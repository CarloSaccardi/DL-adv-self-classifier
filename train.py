
'''
TODO: Optimizer param group : I need to add it and read about it.

'''




import argparse
from email.mime import base
import os
import random
import shutil
import time
import sys
import yaml
import warnings
import wandb

from loss import Loss
import torch
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as torchvision_models
from torchsummary import summary
import numpy as np
import utils
from torch.cuda.amp import autocast# GradScaler
from model import Model
import matplotlib.pyplot as plt
# sys.path.append(os.path.join(os.path.dirname(__file__), "self-classifier")) 
# from src.utils import *

def parser_func():
    '''
    Parse the arguments

    Returns:
        args: The arguments
    '''
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Self-Supervised Training')
    parser.add_argument('--data', metavar='DIR',
                        help='path to dataset', default="./tiny_dataset")
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18')
    parser.add_argument('--epochs', default=800, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=2, type=int,
                        metavar='N')
    parser.add_argument('--lr', '--learning-rate', default=4.8, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--cos', action='store_true',
                        help='use cosine lr schedule')
    parser.add_argument('--sgd', action='store_true',
                        help='use SGD optimizer')
    parser.add_argument('--lars', action='store_true',
                        help='use LARS optimizer')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-6, type=float,
                        metavar='W', help='weight decay (default: 1e-6)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=16, type=int,
                        metavar='N', help='print frequency (default: 16)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed', default=0, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--cls-size', type=int, default=[5], nargs='+',
                        help='size of classification layer. can be a list if cls-size > 1')
    parser.add_argument('--num-cls', default=1, type=int, metavar='NCLS',
                        help='number of classification layers')
    parser.add_argument('--save-path', default='./saved/', type=str,
                        help='save path for checkpoints')
    parser.add_argument('--pretrained', default=None, type=str,
                        help='path to pretrained checkpoint')
    parser.add_argument('--dim', default=128, type=int, metavar='DIM',
                        help='size of MLP embedding layer')
    parser.add_argument('--hidden-dim', default=4096, type=int, metavar='HDIM',
                        help='size of MLP hidden layer')
    parser.add_argument('--num-hidden', default=3, type=int,
                        help='number of MLP hidden layers')
    parser.add_argument('--use-amp', default = True, action='store_true',
                        help='use automatic mixed precision')
    parser.add_argument('--use-bn', action='store_true',
                        help='use batch normalization layers in MLP')
    parser.add_argument('--fixed-cls', action='store_true',
                        help='use a fixed classifier')
    parser.add_argument('--no-leaky', action='store_true',
                        help='use regular relu layers instead of leaky relu in MLP')
    parser.add_argument('--row-tau', default=0.1, type=float,
                    help='row softmax temperature (default: 0.1)')
    parser.add_argument('--col-tau', default=0.05, type=float,
                    help='column softmax temperature (default: 0.05)')
    parser.add_argument('--eps', type=float, default=1e-12,
                    help='small value to avoid division by zero and log(0)')
    parser.add_argument('--final-lr', default=None, type=float,
                    help='final learning rate (None for constant learning rate)')
    parser.add_argument('--warmup-epochs', default=10, type=int,
                    help='linear warmup epochs (default: 10)')
    parser.add_argument('--start-warmup', default=0.3, type=float,
                    help='initial warmup learning rate')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
    parser.add_argument('--global-crops-scale', type=float, nargs='+', default=(0.4, 1.),
                        help="""Scale range of the cropped image before resizing, relatively to the origin image.
                        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we 
                        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local-crops-number', type=int, default=6,
                        help="""Number of small local views to generate. 
                        Set this parameter to 0 to disable multi-crop training. 
                        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local-crops-scale', type=float, nargs='+', default=(0.05, 0.4),
                        help="""Scale range of the cropped image before resizing, relatively to the origin image. 
                        Used for small local view cropping of multi-crop.""")
    parser.add_argument('--config', type=float, nargs='+', default=(0.05, 0.4),
                        help="""Scale range of the cropped image before resizing, relatively to the origin image. 
                        Used for small local view cropping of multi-crop.""")
    parser.add_argument("--local_config", default=None, help="config path")
    parser.add_argument("--wandb", default=None, help="Specify project name to log using WandB")

    args = parser.parse_args()
    
    return args

def main(args):
    print(args)
    
    
    base_model = torchvision_models.__dict__[args.arch]()
    backbone_dim = base_model.fc.weight.shape[1]
    
    model = Model(base_model=base_model,
                      dim=args.dim,
                      hidden_dim=args.hidden_dim,
                      cls_size=args.cls_size,
                      num_cls=args.num_cls,
                      num_hidden=args.num_hidden,
                      use_bn=args.use_bn,
                      backbone_dim=backbone_dim,
                      fixed_cls=args.fixed_cls,
                      no_leaky=args.no_leaky)

    if args.sgd:
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), args.lr,
                                      weight_decay=args.weight_decay)


    traindir = os.path.join(args.data, 'train')
    transform = utils.DataAugmentation(args.global_crops_scale, args.local_crops_scale, args.local_crops_number)
    dataset = utils.ImageFolderWithIndices(traindir, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    criterion = Loss(row_tau=args.row_tau, col_tau=args.col_tau, eps=args.eps)
    
    # schedulers
    lr_schedule = utils.cosine_scheduler_with_warmup(base_value=args.lr,
                                                     final_value=args.final_lr,
                                                     epochs=args.epochs,
                                                     niter_per_ep=len(loader),
                                                     warmup_epochs=args.warmup_epochs,
                                                     start_warmup_value=args.start_warmup)

    #scaler = GradScaler(enabled=args.use_amp, init_scale=2. ** 14)

    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        loss_i, acc1 = train(loader, model, criterion, optimizer, lr_schedule, epoch, args)#add scaler as input when using GPU
        wandb.log({"train_loss": loss_i, "train_acc": acc1})
        print("##################################################")
        print("train_loss: ", loss_i)
        print("train_acc: ", acc1)
        print("##################################################")
        # remember best acc@1 and save checkpoint
        is_best = True if epoch == 0 else loss_i < best_loss
        best_loss = loss_i if epoch == 0 else min(loss_i, best_loss)

        
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer': optimizer.state_dict(),
        }, is_best=is_best, is_milestone=(epoch + 1) % 25 == 0,
            filename=os.path.join(args.save_path, 'model_last.pth.tar'))





def train(loader, model, criterion, optimizer, lr_schedule, epoch, args):#add scaler as input when using GPU
    
    batch_time = utils.AverageMeter('Time', ':6.3f')
    data_time = utils.AverageMeter('Data', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    progress = utils.ProgressMeter(len(loader), [batch_time, data_time, losses, top1],
                                   prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    
    for i, (images, target, indices) in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)
        """        
        for i in range(len(images)):
            img1 = images[i][1].permute(1, 2, 0).detach().cpu().numpy()
            plt.imshow(img1)
            plt.savefig(str(i)+"test.png")
        """
        # adjust learning rate
        utils.adjust_lr(optimizer, lr_schedule, iteration=epoch * len(loader) + i)
        optimizer.zero_grad()

        # compute output
        with autocast(enabled=args.use_amp):

            embds = model(images, return_embds=True)

            probs = model(embds, return_embds=False)
            with autocast(enabled=False):
                # compute loss
                loss = criterion(probs)

        # measure accuracy and record loss
        #acc1 = utils.accuracy(output, target, topk=(1,))
        losses.update(loss.item(), probs[0][0].size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        #top1.update(acc1[0], images.size(0))

        # compute gradient and do SGD step
        # scaler.scale(loss).backward()       
        # scaler.step(optimizer)
        # scaler.update() 
        
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    return losses.avg, top1.avg

def save_checkpoint(state, is_best, is_milestone, filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(os.path.split(filename)[0], 'model_best.pth.tar'))
        print('Best model was saved.')
    if is_milestone:
        shutil.copyfile(filename, os.path.join(os.path.split(filename)[0], 'model_{}.pth.tar'.format(state['epoch'])))
        print('Milestone {} model was saved.'.format(state['epoch']))

def update_args(args, config_dict):
    for key, val in config_dict.items():
        setattr(args, key, val)

if __name__ == '__main__':
    
    args = parser_func()
    
    if args.local_config is not None:
        with open(str(args.local_config), "r") as f:
            config = yaml.safe_load(f)
        update_args(args, config)
        if args.wandb:
            wandb_config = vars(args)
            run = wandb.init(project=str(args.wandb), entity="self-classifier", config=wandb_config)
            # update_args(args, dict(run.config))
    else:
        warnings.warn("No config file was provided. Using default parameters.")
    main(args)

