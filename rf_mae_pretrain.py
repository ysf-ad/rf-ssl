#!/usr/bin/env python3
"""
RF Signal MAE Pre-training Script

Adapted from Meta's MAE implementation for RF signal spectrograms.
This script pretrains a Masked Autoencoder on RF signal data.
"""

import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms

import sys
# Add mae directory to path for imports
mae_path = os.path.join(os.path.dirname(__file__), 'mae')
sys.path.insert(0, mae_path)

# Import MAE modules
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import models_mae
from engine_pretrain import train_one_epoch

# Import our custom dataset
from rf_dataset import create_rf_dataloaders


def get_args_parser():
    parser = argparse.ArgumentParser('RF Signal MAE pre-training', add_help=False)
    
    # Training parameters
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='preprocessed-data', type=str,
                        help='path to preprocessed RF signal data')
    parser.add_argument('--train_ratio', default=0.8, type=float,
                        help='ratio of data to use for training')

    # Output and logging
    parser.add_argument('--output_dir', default='./rf_mae_output',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./rf_mae_logs',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # Distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # Create RF signal dataloaders
    print(f"Loading RF signal data from {args.data_path}")
    train_loader, val_loader, class_info = create_rf_dataloaders(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_ratio=args.train_ratio,
        pin_memory=args.pin_mem
    )
    
    print(f"Dataset info: {class_info['num_classes']} classes")
    print(f"Classes: {list(class_info['class_to_idx'].keys())}")

    # Setup distributed training if needed
    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        if hasattr(train_loader.dataset, 'samples'):
            # Create a simple distributed sampler for our custom dataset
            sampler_train = torch.utils.data.DistributedSampler(
                train_loader.dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
            # Recreate dataloader with distributed sampler
            train_loader = torch.utils.data.DataLoader(
                train_loader.dataset, sampler=sampler_train,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=True,
            )
        print("Sampler_train = %s" % str(train_loader.sampler))
    else:
        sampler_train = torch.utils.data.RandomSampler(train_loader.dataset)

    # Setup logging
    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    # Define the model
    print(f"Creating model: {args.model}")
    model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)
    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    # Calculate effective batch size and learning rate
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    # Setup distributed training
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    
    # Setup optimizer
    try:
        import timm.optim.optim_factory as optim_factory
        param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    except (ImportError, AttributeError):
        # Fallback for newer timm versions
        param_groups = [{'params': model_without_ddp.parameters(), 'weight_decay': args.weight_decay}]
    
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    # Load checkpoint if resuming
    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    # Create output directory
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Save dataset info and args
        if global_rank == 0:
            with open(os.path.join(args.output_dir, 'class_info.json'), 'w') as f:
                json.dump(class_info, f, indent=2)
            
            with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
                json.dump(vars(args), f, indent=2)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed and hasattr(train_loader, 'sampler'):
            train_loader.sampler.set_epoch(epoch)
            
        train_stats = train_one_epoch(
            model, train_loader,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        
        # Save checkpoint
        if args.output_dir and (epoch % 20 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        # Log stats
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch,}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    
    # Set default paths if not provided
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.log_dir:
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)
        
    main(args) 