from __future__ import print_function
import argparse
import os
import sys
import shutil
import time
from tqdm import tqdm
from pprint import pprint

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from pprint import pprint
from torch.autograd import Variable
import pandas as pd
import numpy as np
import random
from warnings import warn
from torch.multiprocessing import Process
import random
from functools import partial
from datetime import datetime

from wideresnet import WideResNet

from pytorch_ps_mpi import MPI_PS
import codings

today_datetime = datetime.now().isoformat()[:10]
today = '2018-01-26'
if today != today_datetime:
    warn('Is today set correctly?')

# used for logging to TensorBoard
if False:
    from tensorboard_logger import configure, log_value

use_cuda = torch.cuda.is_available()
parser = argparse.ArgumentParser(description='PyTorch WideResNet Training')
parser.add_argument('--dataset', default='cifar10', type=str,
                    help='dataset (cifar10 [default] or cifar100)')
parser.add_argument('--epochs', default=200, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=1024, type=int,
                    help='mini-batch size (default: 512)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='initial learning rate. tuned')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--layers', default=10, type=int,
                    help='total number of layers (default: 28)')
parser.add_argument('--widen-factor', default=1, type=int,
                    help='widen factor (default: 10)')
parser.add_argument('--droprate', default=0, type=float,
                    help='dropout probability (default: 0.0)')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='whether to use standard augmentation (default: True)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='WideResNet-28-10', type=str,
                    help='name of experiment')
parser.add_argument('--tensorboard', default=False,
                    help='Log progress to TensorBoard', action='store_true')
parser.add_argument('--num_workers', default=1, help='Number of workers',
                    type=int)
parser.add_argument('--seed', default=42, help='Random seed', type=int)
parser.add_argument('--compress', default=1,
                    help='Boolean int: compress or not', type=int)
parser.add_argument('--svd_rescale', default=1,
                    help='Boolean int: compress or not', type=int)
parser.add_argument('--svd_rank', default=0, help='Boolean int: compress or not',
                    type=int)
parser.add_argument('--device', default=0, help='Which GPU to use', type=int)
parser.add_argument('--qsgd', default=0, type=int, help='Use QSGD?')
parser.add_argument('--use_mpi', default=1, type=int, help='Use MPI?')

parser.set_defaults(augment=True)
args = parser.parse_args()
args.use_cuda = use_cuda
args.compress = bool(args.compress)
args.qsgd = bool(args.qsgd)
args.use_mpi = bool(args.use_mpi)
args.svd_rescale = bool(args.svd_rescale)
print("args.compress ==", args.compress)
print("args.use_cuda ==", args.use_cuda)


def _set_visible_gpus(num_gpus, device=None, verbose=True):
    gpus = list(range(torch.cuda.device_count()))
    if device is not None:
        devices = [device]
    else:
        devices = gpus[:num_gpus]
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(g) for g in devices])
    if verbose:
        print("CUDA_VISIBLE_DEVICES={}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    return devices

#  device_ids = _set_visible_gpus(args.num_workers, random_gpu=args.random_gpu)
device_ids = _set_visible_gpus(args.num_workers, device=args.device)
cuda_kwargs = {'async': True}

from mpi4py import MPI
size = MPI.COMM_WORLD.Get_size()
#  args.num_workers = max(args.num_workers, size)
args.world_size = size
args.batch_size = args.batch_size // args.world_size
print(args.world_size, args.batch_size)
def _set_seed(seed):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
_set_seed(args.seed)

best_prec1 = 0


def _mkdir(dir):
    if not os.path.isdir(dir):
        os.mkdir(dir)
    return True

def _write_csv(df, id=''):
    filename = f'output/{today}/{id}.csv'
    _mkdir('output')
    _mkdir(f'output/{today}')
    df.to_csv(filename)
    return True

def main():
    global args, best_prec1, data
    if args.tensorboard: configure("runs/%s"%(args.name))

    # Data loading code
    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x/255.0 for x in [63.0, 62.1, 66.7]])

    if args.augment:
        transform_train = transforms.Compose([
        	transforms.ToTensor(),
        	transforms.Lambda(lambda x: F.pad(
        						Variable(x.unsqueeze(0), requires_grad=False, volatile=True),
        						(4,4,4,4),mode='reflect').data.squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            normalize,
            ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
        ])

    # create model
    print("Creating the model...")
    model = WideResNet(args.layers, args.dataset == 'cifar10' and 10 or 100,
                       args.widen_factor, dropRate=args.droprate)

    rank = MPI.COMM_WORLD.Get_rank()
    args.rank = rank
    args.seed += rank
    _set_seed(args.seed)

    print("Creating the DataLoader...")
    kwargs = {'num_workers': args.num_workers}#, 'pin_memory': args.use_cuda}
    assert(args.dataset == 'cifar10' or args.dataset == 'cifar100')
    train_loader = torch.utils.data.DataLoader(
        datasets.__dict__[args.dataset.upper()]('../data', train=True,
                                                download=True,
                                                transform=transform_train),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        datasets.__dict__[args.dataset.upper()]('../data', train=False, transform=transform_test),
        batch_size=args.batch_size, shuffle=False, **kwargs)

    # get the number of model parameters
    args.num_parameters = sum([p.data.nelement() for p in model.parameters()])
    print('Number of model parameters: {}'.format(args.num_parameters))

    # for training on multiple GPUs.
    # Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
    if args.use_cuda:
        print("Moving the model to the GPU")
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        model = model.cuda()
        #model = model.cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # define loss function (criterion) and optimizer

    criterion = nn.CrossEntropyLoss()
    if args.use_cuda:
        criterion = criterion.cuda()
    #  optimizer = torch.optim.SGD(model.parameters(), args.lr,
                              #  momentum=args.momentum, nesterov=args.nesterov,
                              #  weight_decay=args.weight_decay)
    #  optimizer = torch.optim.ASGD(model.parameters(), args.lr)
    #  from distributed_opt import MiniBatchSGD
    #  import torch.distributed as dist
    #  rank = np.random.choice('gloo')
    print('initing MiniBatchSGD')
    print(list(model.parameters())[6].view(-1)[:3])
    if not args.qsgd:
        encode_kwargs = {'random_sample': args.svd_rescale,
                         'svd_rank': args.svd_rank, 'compress': args.compress}
        code = codings.svd.SVD()
    else:
        encode_kwargs = {}
        code = codings.qsgd.QSGD()

    names = [n for n, p in model.named_parameters()]
    assert len(names) == len(set(names))
    optimizer = MPI_PS(model.named_parameters(), model.parameters(), args.lr,
                       code=code,
                       use_mpi=args.use_mpi, cuda=args.use_cuda)

    data = []
    train_data = []
    train_time = 0
    for epoch in range(args.start_epoch, args.epochs + 1):
        print(f"epoch {epoch}")
        adjust_learning_rate(optimizer, epoch+1)

        # train for one epoch
        start = time.time()
        if epoch >= 0:
            train_d = train(train_loader, model, criterion, optimizer, epoch)
        else:
            train_d = []
        train_time += time.time() - start
        train_data += [dict(datum, **vars(args)) for datum in train_d]

        # evaluate on validation set
        datum = validate(val_loader, model, criterion, epoch)
        train_datum = validate(train_loader, model, criterion, epoch)
        data += [{'train_time': train_time,
                  'whole_train_acc': train_datum['acc_train'],
                  'whole_train_loss': train_datum['loss_train'],
                  'epoch': epoch + 1, **vars(args), **datum}]
        if epoch > 0:
            data[-1]['epoch_train_time'] = data[-1]['train_time'] - data[-2]['train_time']
            for key in train_data[-1]:
                values = [datum[key] for i, datum in enumerate(train_data)]
                if 'time' in key:
                    data[-1]["epoch_" + key] = np.sum(values)
                else:
                    data[-1]["epoch_" + key] = values[0]

        df = pd.DataFrame(data)
        train_df = pd.DataFrame(train_data)
        if True:
            time.sleep(1)
            print('\n\nmin_train_loss', train_datum['loss_train'],
                  train_datum['acc_train'], '\n\n')
            time.sleep(1)
        ids = [str(getattr(args, key)) for key in
               ['layers', 'lr', 'batch_size', 'compress', 'seed', 'num_workers',
                'svd_rank', 'svd_rescale', 'use_mpi', 'qsgd', 'world_size', 'rank']]
        _write_csv(df, id=f'-'.join(ids))
        _write_csv(train_df, id=f'-'.join(ids) + '_train')
        pprint({k: v for k, v in data[-1].items()
                if k in ['svd_rank', 'svd_rescale', 'qsgd', 'compress']})
        pprint({k: v for k, v in data[-1].items()
                if k in ['train_time', 'num_workers', 'loss_test',
                         'acc_test', 'epoch', 'compress', 'svd_rank', 'qsgd'] or 'time' in k})
        prec1 = datum['acc_test']

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best)
    print('Best accuracy: ', best_prec1)

def train(train_loader, model, criterion, optimizer, epoch):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    pbar = tqdm(enumerate(train_loader), leave=True)
    comm_data = []
    start = time.time()
    for i, (input, target) in pbar:
        if args.use_cuda:
            target = target.cuda(**cuda_kwargs)
            input = input.cuda(**cuda_kwargs)
        if i > 3:
            break
        if i > 50e3 / 1024:
            break
        print(i)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss_start = time.time()
        loss.backward()
        loss_datum = {'grad_compute_time': time.time() - loss_start}

        r = optimizer.step()
        if r is not None:
            _, comm_datum = r
        else:
            comm_datum = {}
        pprint({**comm_datum, **loss_datum})
        comm_data += [{'loss_train_avg': losses.avg, 'loss_train': losses.val,
                       'acc_train_avg': top1.avg,   'acc_train': top1.val,
                       **comm_datum, **loss_datum}]

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        p = 100 * i / len(train_loader)
        pbar.set_description(f'loss={losses.avg:.3f}, acc={top1.avg:.3f} {p:.1f}%')

    # log to TensorBoard
    if args.tensorboard:
        log_value('train_loss', losses.avg, epoch)
        log_value('train_acc', top1.avg, epoch)
    return comm_data

def validate(val_loader, model, criterion, epoch):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    pbar = tqdm(enumerate(val_loader), leave=True)
    for i, (input, target) in pbar:
        if args.use_cuda:
            target = target.cuda(**cuda_kwargs)
            input = input.cuda(**cuda_kwargs)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        p = 100 * i / len(val_loader)
        pbar.set_description(f'loss={losses.avg:.3f}, acc={top1.avg:.3f} {p:.1f}%')

    #print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    # log to TensorBoard
    if args.tensorboard:
        log_value('val_loss', losses.avg, epoch)
        log_value('val_acc', top1.avg, epoch)
    return {'loss_test': losses.avg,
            'acc_test': top1.avg}


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = "runs/%s/"%(args.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/%s/'%(args.name) + 'model_best.pth.tar')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    #  lr = args.lr * ((0.2 ** int(epoch >= 60)) * (0.2 ** int(epoch >= 120))* (0.2 ** int(epoch >= 160)))
    lr = args.lr
    if epoch % 3 == 0 and epoch > 1:
        lr = args.lr * 0.97**(epoch / 3)
    # log to TensorBoard
    if args.tensorboard:
        log_value('learning_rate', lr, epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1 / batch_size))
    return res


if __name__ == '__main__':
    print("In train.py")
    main()
