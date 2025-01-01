import torch
from torch import nn
import actionllm
from typing import Optional, Tuple
from  torch.cuda.amp import autocast
import actionllm.eval_model


class RepAdapter_Router(nn.Module):
    """ Pytorch Implemention of RepAdapter for 1d tensor"""

    def __init__(
            self,
            in_features=768,
            hidden_dim=8,
            groups=2,
            scale=1,
            t=10.
    ):
        super().__init__()
        self.conv_A=nn.Conv1d(in_features,hidden_dim,1,groups=1,bias=True)
        self.conv_B = nn.Conv1d(hidden_dim, in_features, 1, groups=groups, bias=True)

        self.conv_D = nn.Conv1d(hidden_dim, in_features, 1, groups=groups, bias=True)

        self.expert_weights=nn.Linear(in_features,2)

        self.dropout=nn.Dropout(0.1)
        self.groups=groups
        self.scale=scale
        self.t=t

        nn.init.xavier_uniform_( self.conv_A.weight)
        nn.init.zeros_(self.conv_A.bias)
        nn.init.zeros_(self.conv_B.weight)
        nn.init.zeros_(self.conv_B.bias)


        nn.init.zeros_(self.conv_D.weight)
        nn.init.zeros_(self.conv_D.bias)

    def forward(self, x,weights=None):
        with autocast():
            if weights is None:
                weights=torch.softmax(self.expert_weights(x[:,0])/self.t,-1).half()
            x=x.transpose(1,2)
            x_=self.dropout(self.conv_A(x))
            # x=self.conv_B(x_)*self.scale*weights[:,0,None,None]+self.conv_D(x_)*self.scale*weights[:,1,None,None]+x
            x = self.conv_D(x_)+x #FY0829
            x=x.transpose(1,2).contiguous()
        return x

def forward_llama_attn(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], adapter=None):
    if self.training and self.gradient_checkpointing:
        h = x + self.drop_path(torch.utils.checkpoint.checkpoint(self.attention, self.adapter_attn(self.attention_norm(x)), start_pos, freqs_cis, mask))
        out = h + self.drop_path(torch.utils.checkpoint.checkpoint(self.feed_forward, self.ffn_norm(h)))
    else:
        h = x + self.drop_path(self.attention.forward(self.adapter_attn(self.attention_norm(x)), start_pos, freqs_cis, mask, adapter))
        out = h + self.drop_path(self.feed_forward.forward(self.ffn_norm(h)))
    return out


def set_MMAdapter(model, method, dim=8, s=1, set_forward=True,t=10,gradient_checkpointing=False):
    # adapter_single = RepAdapter_Router(4096, hidden_dim=dim, scale=s, t=t)    # TY0110
    if method == 'attn':
        for _ in model.children():
            if type(_) == actionllm.model.TransformerBlock:
                # _.adapter_attn = adapter_single     # TY 0110
                _.adapter_attn = RepAdapter_Router(_.dim,hidden_dim=dim,scale=s,t=t)  # TY0110
                _.s = s
                _.t=t
                _.gradient_checkpointing = gradient_checkpointing
                if type(_) == actionllm.model.TransformerBlock:
                    bound_method = forward_llama_attn.__get__(_, _.__class__)
                if set_forward:
                    setattr(_, 'forward', bound_method)
            elif len(list(_.children())) != 0:
                set_MMAdapter(_, method, dim, s, set_forward=set_forward,t=t,gradient_checkpointing=gradient_checkpointing)

