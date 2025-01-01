# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Optional, Tuple
from dataclasses import dataclass
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Embedding, Linear
from timm.models.layers import  DropPath
import clip
from torch.cuda.amp import autocast
import torch.nn as nn
from util.action_tool import normalize_duration
from torch.nn import Dropout, Softmax

@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5
    hidden_proj: int=128
    feature_dim: int=2048
    num_class: int=48
    n_query: int=8
    multi_hidden_proj :int=64

    max_batch_size: int = 32
    max_seq_len: int = 2048
    drop_path: float = 0.
    drop_out_rate: float = 0.1




class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_local_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads

        #modified bias for reparameterizing
        self.wq = Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False
        )
        self.wk = Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False
        )
        self.wv = Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False
        )
        self.wo = Linear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False
        )

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], adapter=None):

        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        keys = xk
        values = xv

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, slen, head_dim)
        output = output.transpose(
            1, 2
        ).contiguous().view(bsz, seqlen, -1)

        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = Linear(
            dim, hidden_dim, bias=False
        )
        self.w2 = Linear(
            hidden_dim, dim, bias=False
        )
        self.w3 = Linear(
            dim, hidden_dim, bias=False
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x),inplace=False) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim, hidden_dim=4 * args.dim, multiple_of=args.multiple_of
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.drop_path = DropPath(args.drop_path) if args.drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], adapter=None):

        h = x + self.drop_path(self.attention.forward(self.attention_norm(x), start_pos, freqs_cis, mask, adapter))
        out = h + self.drop_path(self.feed_forward.forward(self.ffn_norm(h)))
        return out


class AdapterMLP(nn.Module):
    """ Pytorch Implemention of RepAdapter for 1d tensor"""

    def __init__(
            self,
            in_features=768,
            hidden_dim=128,
            out_features=4096
    ):
        super().__init__()
        self.conv_A=nn.Linear(in_features,hidden_dim)
        self.conv_B = nn.Linear(hidden_dim, out_features)

        nn.init.xavier_uniform_( self.conv_A.weight)
        nn.init.zeros_(self.conv_A.bias)
        nn.init.xavier_uniform_(self.conv_B.weight)
        nn.init.zeros_(self.conv_B.bias)

    def forward(self, x):
        with autocast():
            x=self.conv_B(F.silu(self.conv_A(x)))
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x
class Multi_Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super(Multi_Attention, self).__init__()
        self.num_attention_heads = args.n_heads
        self.attention_head_size = int(args.multi_hidden_proj / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # text
        self.query_text = Linear(args.multi_hidden_proj, self.all_head_size)
        self.key_text = Linear(args.multi_hidden_proj, self.all_head_size)
        self.value_text = Linear(args.multi_hidden_proj, self.all_head_size)
        self.out_text = Linear(args.multi_hidden_proj, args.multi_hidden_proj)
        self.proj_dropout_text = Dropout(args.drop_out_rate)
        self.attn_dropout_text = Dropout(args.drop_out_rate)

        # visual
        self.query_visual = Linear(args.multi_hidden_proj, self.all_head_size)
        self.key_visual = Linear(args.multi_hidden_proj, self.all_head_size)
        self.value_visual = Linear(args.multi_hidden_proj, self.all_head_size)
        self.out_visual = Linear(args.multi_hidden_proj, args.multi_hidden_proj)
        self.proj_dropout_visual = Dropout(args.drop_out_rate)
        self.attn_dropout_visual = Dropout(args.drop_out_rate)

        # query
        self.query_Q = Linear(args.multi_hidden_proj, self.all_head_size)
        self.key_Q = Linear(args.multi_hidden_proj, self.all_head_size)
        self.value_Q = Linear(args.multi_hidden_proj, self.all_head_size)
        self.out_Q = Linear(args.multi_hidden_proj, args.multi_hidden_proj)
        self.proj_dropout_Q = Dropout(args.drop_out_rate)
        self.attn_dropout_Q = Dropout(args.drop_out_rate)

        # cross
        self.attn_dropout_it = Dropout(args.drop_out_rate)
        self.attn_dropout_iQ = Dropout(args.drop_out_rate)

        self.attn_dropout_ti = Dropout(args.drop_out_rate)
        self.attn_dropout_tQ = Dropout(args.drop_out_rate)

        self.attn_dropout_Qi = Dropout(args.drop_out_rate)
        self.attn_dropout_Qt = Dropout(args.drop_out_rate)

        self.softmax = Softmax(dim=-1)
    def transpose_for_scores(self, x):
        # print(self.num_attention_heads, self.attention_head_size)
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, text, visual, query):

        # Get q, k, v corresponding to the three modes in attention
        text_q = self.query_text(text)
        text_k = self.key_text(text)
        text_v = self.value_text(text)

        visual_q = self.query_visual(visual)
        visual_k = self.key_visual(visual)
        visual_v = self.value_visual(visual)

        Q_q = self.query_Q(query)
        Q_k = self.key_Q(query)
        Q_v = self.value_Q(query)

        # Dimension conversion
        query_layer_text = self.transpose_for_scores(text_q)
        key_layer_text = self.transpose_for_scores(text_k)
        value_layer_text = self.transpose_for_scores(text_v)

        query_layer_vis = self.transpose_for_scores(visual_q)
        key_layer_vis = self.transpose_for_scores(visual_k)
        value_layer_vis = self.transpose_for_scores(visual_v)

        query_layer_Q = self.transpose_for_scores(Q_q)
        key_layer_Q = self.transpose_for_scores(Q_k)
        value_layer_Q = self.transpose_for_scores(Q_v)

        # attention compute
        # q*k
        attention_scores_vis = torch.matmul(query_layer_vis, key_layer_vis.transpose(-1, -2))
        attention_scores_text = torch.matmul(query_layer_text, key_layer_text.transpose(-1, -2))
        attention_scores_Q = torch.matmul(query_layer_Q, key_layer_Q.transpose(-1, -2))

        attention_scores_it = torch.matmul(query_layer_vis, key_layer_text.transpose(-1, -2))
        attention_scores_iQ = torch.matmul(query_layer_vis, key_layer_Q.transpose(-1, -2))
        attention_scores_ti = torch.matmul(query_layer_text, key_layer_vis.transpose(-1, -2))
        attention_scores_tQ = torch.matmul(query_layer_text, key_layer_Q.transpose(-1, -2))
        attention_scores_Qi = torch.matmul(query_layer_Q, key_layer_vis.transpose(-1, -2))
        attention_scores_Qt = torch.matmul(query_layer_Q, key_layer_text.transpose(-1, -2))

        # Attention probability distribution
        attention_scores_text = attention_scores_text / math.sqrt(self.attention_head_size)
        attention_probs_text = self.softmax(attention_scores_text)
        attention_probs_text = self.attn_dropout_text(attention_probs_text)

        attention_scores_vis = attention_scores_vis / math.sqrt(self.attention_head_size)
        attention_probs_vis = self.softmax(attention_scores_vis)
        attention_probs_vis = self.attn_dropout_visual(attention_probs_vis)

        attention_scores_Q = attention_scores_Q / math.sqrt(self.attention_head_size)
        attention_probs_Q = self.softmax(attention_scores_Q)
        attention_probs_Q = self.attn_dropout_Q(attention_probs_Q)


        attention_scores_it = attention_scores_it / math.sqrt(self.attention_head_size)
        attention_probs_it = self.softmax(attention_scores_it)
        attention_probs_it = self.attn_dropout_it(attention_probs_it)

        attention_scores_iQ = attention_scores_iQ / math.sqrt(self.attention_head_size)
        attention_probs_iQ = self.softmax(attention_scores_iQ)
        attention_probs_iQ = self.attn_dropout_iQ(attention_probs_iQ)

        attention_scores_ti = attention_scores_ti / math.sqrt(self.attention_head_size)
        attention_probs_ti = self.softmax(attention_scores_ti)
        attention_probs_ti = self.attn_dropout_ti(attention_probs_ti)

        attention_scores_tQ = attention_scores_tQ / math.sqrt(self.attention_head_size)
        attention_probs_tQ = self.softmax(attention_scores_tQ)
        attention_probs_tQ = self.attn_dropout_tQ(attention_probs_tQ)

        attention_scores_Qi = attention_scores_Qi / math.sqrt(self.attention_head_size)
        attention_probs_Qi = self.softmax(attention_scores_Qi)
        attention_probs_Qi = self.attn_dropout_Qi(attention_probs_Qi)

        attention_scores_Qt = attention_scores_Qt / math.sqrt(self.attention_head_size)
        attention_probs_Qt = self.softmax(attention_scores_Qt)
        attention_probs_Qt = self.attn_dropout_Qt(attention_probs_Qt)

        # The context vector obtained by multiplying the attention probability with the corresponding value matrix
        context_layer_vis = torch.matmul(attention_probs_vis, value_layer_vis)
        context_layer_vis = context_layer_vis.permute(0, 2, 1, 3).contiguous()
        context_layer_text = torch.matmul(attention_probs_text, value_layer_text)
        context_layer_text = context_layer_text.permute(0, 2, 1, 3).contiguous()
        context_layer_Q = torch.matmul(attention_probs_Q, value_layer_Q)
        context_layer_Q = context_layer_Q.permute(0, 2, 1, 3).contiguous()

        context_layer_it = torch.matmul(attention_probs_it, value_layer_text)
        context_layer_it = context_layer_it.permute(0, 2, 1, 3).contiguous()
        context_layer_iQ = torch.matmul(attention_probs_iQ, value_layer_Q)
        context_layer_iQ = context_layer_iQ.permute(0, 2, 1, 3).contiguous()

        context_layer_ti = torch.matmul(attention_probs_ti, value_layer_vis)
        context_layer_ti = context_layer_ti.permute(0, 2, 1, 3).contiguous()
        context_layer_tQ = torch.matmul(attention_probs_tQ, value_layer_Q)
        context_layer_tQ = context_layer_tQ.permute(0, 2, 1, 3).contiguous()

        context_layer_Qi = torch.matmul(attention_probs_Qi, value_layer_vis)
        context_layer_Qi = context_layer_Qi.permute(0, 2, 1, 3).contiguous()
        context_layer_Qt = torch.matmul(attention_probs_Qt, value_layer_text)
        context_layer_Qt = context_layer_Qt.permute(0, 2, 1, 3).contiguous()

        # reshape for output
        new_context_layer_shape = context_layer_text.size()[:-2] + (self.all_head_size,)
        context_layer_text = context_layer_text.view(*new_context_layer_shape)
        new_context_layer_shape = context_layer_vis.size()[:-2] + (self.all_head_size,)
        context_layer_vis = context_layer_vis.view(*new_context_layer_shape)
        new_context_layer_shape = context_layer_Q.size()[:-2] + (self.all_head_size,)
        context_layer_Q = context_layer_Q.view(*new_context_layer_shape)

        new_context_layer_shape = context_layer_it.size()[:-2] + (self.all_head_size,)
        context_layer_it = context_layer_it.view(*new_context_layer_shape)
        new_context_layer_shape = context_layer_iQ.size()[:-2] + (self.all_head_size,)
        context_layer_iQ = context_layer_iQ.view(*new_context_layer_shape)

        new_context_layer_shape = context_layer_ti.size()[:-2] + (self.all_head_size,)
        context_layer_ti = context_layer_ti.view(*new_context_layer_shape)
        new_context_layer_shape = context_layer_tQ.size()[:-2] + (self.all_head_size,)
        context_layer_tQ = context_layer_tQ.view(*new_context_layer_shape)

        new_context_layer_shape = context_layer_Qi.size()[:-2] + (self.all_head_size,)
        context_layer_Qi = context_layer_Qi.view(*new_context_layer_shape)
        new_context_layer_shape = context_layer_Qt.size()[:-2] + (self.all_head_size,)
        context_layer_Qt = context_layer_Qt.view(*new_context_layer_shape)


        # output
        attention_output_text = self.out_text((context_layer_text + context_layer_ti + context_layer_tQ) / 3)
        attention_output_vis = self.out_visual((context_layer_vis + context_layer_it + context_layer_iQ) / 3)
        attention_output_Q = self.out_Q((context_layer_Q + context_layer_Qt + context_layer_Qi) / 3)

        attention_output_text = self.proj_dropout_text(attention_output_text)
        attention_output_vis = self.proj_dropout_visual(attention_output_vis)
        attention_output_Q = self.proj_dropout_Q(attention_output_Q)


        return attention_output_text, attention_output_vis, attention_output_Q


class Multimodal_Attention_Block(nn.Module):
    def __init__(self, args: ModelArgs):
        super(Multimodal_Attention_Block, self).__init__()
        self.ffn_visual = FeedForward(
            dim=args.multi_hidden_proj, hidden_dim=4 * args.multi_hidden_proj, multiple_of=args.multiple_of
        )
        self.ffn_text = FeedForward(
            dim=args.multi_hidden_proj, hidden_dim=4 * args.multi_hidden_proj, multiple_of=args.multiple_of
        )
        self.ffn_query = FeedForward(
            dim=args.multi_hidden_proj, hidden_dim=4 * args.multi_hidden_proj, multiple_of=args.multiple_of
        )

        self.attn_visual_norm = RMSNorm(args.multi_hidden_proj, eps=args.norm_eps)
        self.attn_text_norm = RMSNorm(args.multi_hidden_proj, eps=args.norm_eps)
        self.attn_query_norm = RMSNorm(args.multi_hidden_proj, eps=args.norm_eps)

        self.ffn_visual_norm = RMSNorm(args.multi_hidden_proj, eps=args.norm_eps)
        self.ffn_text_norm = RMSNorm(args.multi_hidden_proj, eps=args.norm_eps)
        self.ffn_query_norm = RMSNorm(args.multi_hidden_proj, eps=args.norm_eps)

        self.pos_encoder = PositionalEncoding(args.multi_hidden_proj)

        self.attn = Multi_Attention(args)


    def forward(self, text, visual, query):
        text = self.pos_encoder(text)
        visual = self.pos_encoder(visual)
        query = self.pos_encoder(query)

        h_text = text
        h_visual = visual
        h_query = query

        text = self.attn_text_norm(text)
        visual = self.attn_visual_norm(visual)
        query = self.attn_query_norm(query)

        text, visual, query= self.attn(text, visual, query)

        text = text + h_text
        visual = visual + h_visual
        query = query + h_query

        h_text = text
        h_visual = visual
        h_query = query

        text = self.ffn_text_norm(text)
        visual = self.ffn_visual_norm(visual)
        query = self.ffn_query_norm(query)

        text = self.ffn_text(text)
        visual = self.ffn_visual(visual)
        query = self.ffn_query(query)

        text = text + h_text
        visual = visual + h_visual
        query = query + h_query

        return text, visual, query



class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.tok_embeddings = Embedding(
            params.vocab_size, params.dim
        )  # vocab_size: 32000, dim: 4096

        # NOTICE: do not set ignore_index
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = Linear(params.dim, params.vocab_size, bias=False)
        # self.output_cls = Linear(params.dim, params.num_class, bias=False)   # 19 ,48 are number of class in 50s and breakfast dataset

        self.output_seg_text = Linear(params.dim, params.num_class, bias=False)
        self.output_seg_vis = Linear(params.dim, params.num_class, bias=False)

        self.output_action = Linear(params.dim, params.num_class+1,bias=False)  # 19 ,48 are number of class in 50s and breakfast dataset
        self.adapter_output_duration = Linear(params.dim, 1, bias=False)

        self.pad_idx = params.num_class + 1

        self.criterion_seg = torch.nn.CrossEntropyLoss(ignore_index=-100)
        self.criterion_act = torch.nn.CrossEntropyLoss(ignore_index=self.pad_idx)
        self.criterion_dur = torch.nn.MSELoss(reduction='none')

        self.n_query = params.n_query

        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )

        # self.backbone = clip.load('ViT-L/14')[0]
        self.adapter_multi_down = Linear(params.dim, params.multi_hidden_proj, bias=False)
        self.adapter_multi_up = Linear(params.multi_hidden_proj, params.dim, bias=False)
        self.adapter_proj = AdapterMLP(params.feature_dim, params.hidden_proj, params.dim).float()
        # self.padding_feature = nn.Parameter(torch.zeros(610, 2048))   #610*2=1220
        self.adapter_multimodal_attention = Multimodal_Attention_Block(params)

    def pos_emb(self, pos, dim):
        pe=torch.zeros(pos,dim)
        position = torch.arange(pos).unsqueeze(1)   #column vector [pos,1]
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))   #row vector [1,dim//2]
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe

    def trainable_pos_emb(self,pos,dim):
        pe=nn.Embedding(pos,dim)
        nn.init.constant_(pe.weight,0.)

        return pe

    def forward(self, inputs_embeds, text_inputs_embeds, labels_action, past_labels, labels_duration,return_preds=False):
        text_inputs_embeds = text_inputs_embeds.half()

        num_feature = inputs_embeds.size(1)  # 605
        batchsize = inputs_embeds.size(0)

        query = torch.ones(batchsize, self.n_query, inputs_embeds.shape[2]).to(inputs_embeds.device)
        inputs_embeds = self.adapter_proj(inputs_embeds)  # [B, num_feature, 2048] --> [B, num_feature, 4096]
        query = self.adapter_proj(query)  # [B, n_query, 2048] --> [B, n_query, 4096]

        # 4096
        h_query = query
        h_inputs_embeds = inputs_embeds
        h_text_inputs_embeds = text_inputs_embeds

        with autocast():
            query = self.adapter_multi_down(query)
            inputs_embeds = self.adapter_multi_down(inputs_embeds)
            text_inputs_embeds = self.adapter_multi_down(text_inputs_embeds)

            text_inputs_embeds, inputs_embeds, query = self.adapter_multimodal_attention(text_inputs_embeds, inputs_embeds, query)

        # 4096
            text_inputs_embeds = h_text_inputs_embeds + self.adapter_multi_up(text_inputs_embeds)
            inputs_embeds = h_inputs_embeds + self.adapter_multi_up(inputs_embeds)
            query = h_query + self.adapter_multi_up(query)

        inputs_embeds = torch.cat((text_inputs_embeds, inputs_embeds, query), dim=1)
        h = inputs_embeds
        seqlen = inputs_embeds.shape[1]  # [990]

        freqs_cis = self.freqs_cis.to(h.device)  # [2L, 12]
        freqs_cis = freqs_cis[:seqlen]  # [990, 64]

        mask = torch.full((1, 1, seqlen, seqlen), 0., device=h.device)

        start_pos = 0
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)

        h = self.norm(h)  # [B, 990, 4096]
        hidden_states_past_text = h[:, :num_feature, :]
        hidden_states_past_vis = h[:, num_feature:2*num_feature, :]
        hidden_states_future = h[:, 2*num_feature:, :]

        with autocast():
            output_seg_text = self.output_seg_text(hidden_states_past_text)  # [B, num_feature, 4096]-->[B, num_feature, n_class]
            output_seg_vis = self.output_seg_vis(hidden_states_past_vis)  # [B, num_feature, 4096]-->[B, num_feature, n_class]
            output_action = self.output_action(hidden_states_future)  # [B, n_query, 4096]-->[B, n_query, n_class+1]
            output_duration = self.adapter_output_duration(hidden_states_future)  # [B, n_query, 4096]-->[B, n_query, 1]

        output_duration = output_duration.reshape(batchsize, -1)

        if return_preds:
            return {'duration': output_duration, 'action': output_action}

        labels_duration_mask = (labels_duration != self.pad_idx).long().to(inputs_embeds.device)
        labels_duration = labels_duration * labels_duration_mask
        labels_duration = labels_duration.half()

        output_duration = normalize_duration(output_duration, labels_duration_mask)

        num_class = output_seg_text.shape[2]
        output_seg_text = output_seg_text.reshape(-1,num_class)  # [990,19] ; 19 ,48 are number of class in 50s and breakfast dataset
        output_seg_vis = output_seg_vis.reshape(-1, num_class)
        output_action = output_action.reshape(-1, num_class + 1)

        past_labels = past_labels.flatten()  # [605]
        labels_action = labels_action.flatten()

        loss_future_action = self.criterion_act(output_action, labels_action)
        loss_future_duration = torch.sum(self.criterion_dur(output_duration, labels_duration))
        loss_seg_text = self.criterion_seg(output_seg_text, past_labels)
        loss_seg_vis = self.criterion_seg(output_seg_vis, past_labels)

        loss = loss_future_action + loss_future_duration+ loss_seg_text + loss_seg_vis

        return loss