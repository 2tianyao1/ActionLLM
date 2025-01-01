import argparse


def get_args_parser():
    parser = argparse.ArgumentParser('Long_term action anticipation', add_help=False)
    # NOTICE: parameters that should be adjust before running


    parser.add_argument('--epochs', default=80, type=int)    #epoch
    # NOTICE: adjust max_seq_len with the variation of image feature
    parser.add_argument('--max_seq_len', type=int, default=2270, metavar='LENGTH', help='the maximum sequence length')    #max_seq_len
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size per GPU')  # batchsize

    # path
    exp_name = 'test'
    parser.add_argument('--output_dir', default='./result/bf/'+exp_name+'_weights', help='path where to save, empty for no saving') #TY
    parser.add_argument('--log_dir', default='./result/outputs/'+exp_name+'_logs', help='path where to tensorboard log') #TY
    parser.add_argument('--checkpoint_file', type=str,
                        default='/PATH/TO/CHECKPOINT/checkpoint-19.pth')  #checkpoint_file
    parser.add_argument('--pred_file', type=str,
                        default='/path/to/save/file/bf_sp1.xlsx')  #pred_file
    parser.add_argument('--data_root', type=str, default='/path/to/data')  # data_root
    parser.add_argument('--text_feature', type=str, default='/path/to/text/feature')


    parser.add_argument('--accum_iter', default=2, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')


    # If GPU memory is not sufficient, adjust these parameters
    parser.add_argument('--bits', default='16bit', type=str, choices=['4bit', '8bit', '16bit'],
                        help='Quantization bits for training, fp16 by default')
    parser.add_argument('--gradient_checkpointing', default=True,
                        help='saving memory costs via gradient_checkpointing')
    parser.add_argument('--pin_mem', default=False,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')

    # These parameter do not need adjust
    parser.add_argument('--llama_model_path', default='/path/to/weights/7B', type=str, help='path of llama model')   #TY
    parser.add_argument('--llm_model', default='7B', type=str, metavar='MODEL', help='Name of llm model to train')
    # todo: remove distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # adapter
    parser.add_argument('--adapter_type', type=str, default='attn', metavar='LENGTH', choices=['attn'], help='the insert position  of adapter layer')
    parser.add_argument('--visual_adapter_type', type=str, default='router', metavar='LENGTH', choices=['normal', 'router', 'router_block'], help='the type of adapter layer')
    parser.add_argument('--adapter_dim', type=int, default=4, metavar='LENGTH', help='the dims of adapter layer')
    parser.add_argument('--hidden_proj', type=int, default=128, metavar='LENGTH', help='the visual adapter dim')
    parser.add_argument('--temperature', type=float, default=10., metavar='LENGTH', help='the temperature of router')
    parser.add_argument('--adapter_scale', type=float, default=1., metavar='LENGTH', help='the scales of adapter layer')
    parser.add_argument('--drop_path', type=float, default=0., metavar='LENGTH', help='drop path')
    parser.add_argument('--drop_out_rate', type=float, default=0.1, metavar='LENGTH', help='drop out rate')
    parser.add_argument('--multi_hidden_proj', type=int, default=128, metavar='LENGTH', help='the visual adapter dim')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.02, help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate (absolute lr)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='clip gradient', help='clips gradient norm of an iterable of parameters')
    parser.add_argument('--blr', type=float, default=9e-3, metavar='LR', help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=float, default=2, metavar='N', help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')

    # Action anticipation parameters
    parser.add_argument("--sample_rate", type=int, default=8)   # sample_rate
    parser.add_argument("--dataset", type=str, default='50_salads',help='50_salads or breakfast')
    parser.add_argument("--action_class", type=int, default=19)  # It varies with different data sets. If it is breakfast, the value is 19 ,or 48
    parser.add_argument("--split", default="1", help='Different ways of dividing the dataset')   #split
    parser.add_argument("--ck_num", type=int, default=4,help='checkpoint serial number at the time of testing')
    parser.add_argument('--feature_dim', type=int, default=2048, metavar='LENGTH', help='The original dimension of the input feature')
    parser.add_argument("--n_query", type=int, default=20)
    parser.add_argument("--pred_ratio", type=float, default=0.5, help='Percentage of future videos you want to predict')



    # Dora




    return parser