import os
import shutil
import warnings
import numpy as np
import torch as th
import random
import math
import urllib.request
import torch.distributed as dist
from torchvision import transforms, datasets

from PIL import Image, ImageFilter, ImageOps


def cosine_scheduler_with_warmup(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    final_value = base_value if final_value is None else final_value
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule

    
class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(self.radius_min, self.radius_max)))

class ImageFolderWithIndices(datasets.ImageFolder):
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithIndices, self).__getitem__(index)
        # make a new tuple that includes original and the index
        tuple_with_path = (original_tuple + (index,))
        return tuple_with_path

class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class DataAugmentation(object):
    # taken from DINO
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(0.1),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(1.0),
            Solarization(0.2),
            normalize,
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = list()
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops

def save_checkpoint(state, is_best, is_milestone, filename):
    th.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(os.path.split(filename)[0], 'model_best.pth.tar'))
        print('Best model was saved.')
    if is_milestone:
        shutil.copyfile(filename, os.path.join(os.path.split(filename)[0], 'model_{}.pth.tar'.format(state['epoch'])))
        print('Milestone {} model was saved.'.format(state['epoch']))

def accuracy(output, target):
    '''computing accuracy between a list of probabilites for each augmentation and batch with a list of target classes'''
    output = output[0]
    #output shape 8, 2, 5
    #target shape 2, 1
    acc = 0
    for i in range(len(output)):
        acc += (output[i].argmax(dim=1) == target).float().mean()

    return acc / len(output)


class NNQueue:
    def __init__(self, queue_len=131072, dim=128, gpu=None):
        super().__init__()
        self.queue_len = queue_len
        self.dim = dim

        self.queue = th.zeros(self.queue_len, self.dim)
        self.queue_targets = th.zeros(self.queue_len)  # only used for monitoring progress
        self.queue_indices = th.zeros(self.queue_len, dtype=th.long)  # used to avoid choosing the same sample as NN

        if th.cuda.is_available():
            self.queue = self.queue.cuda(gpu, non_blocking=True)
            self.queue_targets = self.queue_targets.cuda(gpu, non_blocking=True)
            self.queue_indices = self.queue_indices.cuda(gpu, non_blocking=True)

        self.ptr = 0
        self.full = False

    def get_nn(self, x, x_indices):
        # extract top2 in case first sample is the query sample itself which can happen
        # in the first few iterations of a new epoch
        _, q_indices = (x @ self.queue.T).topk(2)  # extract indices of queue for top2
        sample_indices = self.queue_indices[q_indices]  # extract 'global' indices of extracted samples
        indices = th.where(x_indices == sample_indices[:, 0], q_indices[:, 1], q_indices[:, 0])

        # extract values
        out = self.queue[indices]
        targets = self.queue_targets[indices]  # only used for monitoring progress, not for training
        return out, targets

    def push(self, x, x_targets, x_indices):
        x_size = x.shape[0]
        old_ptr = self.ptr
        if self.ptr + x_size <= self.queue_len:
            self.queue[self.ptr: self.ptr + x_size] = x
            self.queue_targets[self.ptr: self.ptr + x_size] = x_targets
            self.queue_indices[self.ptr: self.ptr + x_size] = x_indices
            self.ptr = (self.ptr + x_size) % self.queue_len

        else:
            self.queue[self.ptr:] = x[:self.queue_len - old_ptr]
            self.queue_targets[self.ptr:] = x_targets[:self.queue_len - old_ptr]
            self.queue_indices[self.ptr:] = x_indices[:self.queue_len - old_ptr]

            self.ptr = (self.ptr + x_size) % self.queue_len

            self.queue[:self.ptr] = x[self.queue_len - old_ptr:]
            self.queue_targets[:self.ptr] = x_targets[self.queue_len - old_ptr:]
            self.queue_indices[:self.ptr] = x_indices[self.queue_len - old_ptr:]

        if not self.full and old_ptr + x_size >= self.queue_len:
            self.full = True



    

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_lr(optimizer, lr_schedule, iteration):
    for idx, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = lr_schedule[iteration]


