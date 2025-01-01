import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
torch.set_num_threads(4)
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import datetime
import json
import time
import numpy as np
from pathlib import Path
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from engine import train_one_epoch, val_one_epoch

from util.action_dataset import ActionDataset
from actionllm.mm_adaptation import ActionLLM
from util.action_tool import read_mapping_dict, backup_code
from util.opts import get_args_parser

cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
if cuda_visible_devices is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # save code
    # code_version = 'sofar_best_for_bf'
    # backup_code(code_version)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dataset = args.dataset
    split = args.split

    if dataset == 'breakfast':
        data_path = os.path.join(args.data_root,'breakfast')
    elif dataset == '50_salads' :
        data_path = os.path.join(args.data_root,'50_salads')

    mapping_file = os.path.join(data_path, 'mapping.txt')
    actions_dict = read_mapping_dict(mapping_file)
    video_file_path = os.path.join(data_path, 'splits', 'train.split' + split + '.bundle')
    video_file_test_path = os.path.join(data_path, 'splits', 'test.split' + split + '.bundle')

    video_file = open(video_file_path, 'r')
    video_file_test = open(video_file_test_path, 'r')

    video_list = video_file.read().split('\n')[:-1]
    video_test_list = video_file_test.read().split('\n')[:-1]

    features_path = os.path.join(data_path, 'features')
    text_feature_path = os.path.join(args.text_feature, dataset, 'feature_class_result_res50', f"sp{split}")
    gt_path = os.path.join(data_path, 'groundTruth')

    n_class = len(actions_dict) + 1
    pad_idx = n_class

    dataset_train = ActionDataset(args, 'train',video_list,actions_dict,features_path,text_feature_path,gt_path,dataset,n_class,pad_idx)
    dataset_val = ActionDataset(args, 'val',video_test_list,actions_dict,features_path,text_feature_path,gt_path,dataset,n_class,pad_idx)   #small data

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
    sampler_val = torch.utils.data.DistributedSampler(dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)

    print("Sampler_train = %s" % str(sampler_train))
    print("Sampler_val = %s" % str(sampler_val))

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        collate_fn=dataset_train.my_collate
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        collate_fn=dataset_train.my_collate
    )

    # define the model
    model = ActionLLM(args)
    model.to(device)

    model_without_ddp = model

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.weight_decay)

    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))

    # mixed precision scaler
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):

        data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )

        if args.output_dir:
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch }

        # validation
        data_loader_val.sampler.set_epoch(epoch)
        loss_val = val_one_epoch(model, data_loader_val, device)
        print('Val mean loss: ', loss_val.mean())

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + " Val mean loss: " + str(loss_val.mean()) + "\n")


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':

    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
