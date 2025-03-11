from __future__ import print_function

import os
import sys
import argparse
import time
import math

import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets

from util import TwoCropTransform, AverageMeter
from util import adjust_learning_rate, warmup_learning_rate
from util import set_optimizer, save_model
# from networks.resnet_big import WeaklyConResNet
from networks.yolov8 import WeaklyConResNet
# from losses import WeaklyConLoss
from losses import CombinedContrastiveLoss, WeaklyConLoss, DenseContrastiveLoss

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=1,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=1,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=10,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=800,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet')
    parser.add_argument('--dataset', type=str, default='path',
                        choices=['cifar10', 'cifar100', 'path'], help='dataset')
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default='dataset', help='path to custom dataset')
    parser.add_argument('--size', type=int, default=224, help='parameter for RandomResizedCrop')

    # method
    parser.add_argument('--method', type=str, default='WeaklyCon',
                        choices=['WeaklyCon', 'SimCLR'], help='choose method')

    # temperature
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')
    
    parser.add_argument('--loss_type', type=str, default='global', 
                        choices=['global', 'dense', 'combined'], help='choose loss type')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')
    parser.add_argument('--resume', type=str, default='',
                        help='path to latest checkpoint (default: None)')

    opt = parser.parse_args()

    # check if dataset is path that passed required arguments
    if opt.dataset == 'path':
        assert opt.data_folder is not None \
            and opt.mean is not None \
            and opt.std is not None

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = './datasets/'
    opt.model_path = './save_yolov8m/Simclr/{}_models'.format(opt.dataset)
    opt.tb_path = './save_yolov8m/Simclr/{}_tensorboard'.format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}'.\
        format(opt.method, opt.dataset, opt.model, opt.learning_rate,
               opt.weight_decay, opt.batch_size, opt.temp, opt.trial)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    opt.start_epoch = 1  # 初始化 start_epoch

    return opt


def set_loader(opt):
    # construct data loader
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif opt.dataset == 'path':
        mean = eval(opt.mean)
        std = eval(opt.std)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])

    if opt.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                         transform=TwoCropTransform(train_transform),
                                         download=True)
    elif opt.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                          transform=TwoCropTransform(train_transform),
                                          download=True)
    elif opt.dataset == 'path':
        train_dataset = datasets.ImageFolder(root=opt.data_folder,
                                            transform=TwoCropTransform(train_transform))
    else:
        raise ValueError(opt.dataset)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)

    return train_loader


def set_model(opt):
    model = WeaklyConResNet(name=opt.model)
    
    if opt.loss_type == 'global':
        criterion = WeaklyConLoss(temperature=opt.temp)
    elif opt.loss_type == 'dense':
        criterion = DenseContrastiveLoss(temperature=opt.temp)
    else:
        criterion = CombinedContrastiveLoss(global_temperature=opt.temp)

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion


def load_checkpoint(opt, model, optimizer):
    if opt.resume and os.path.isfile(opt.resume):
        print("=> loading checkpoint '{}'".format(opt.resume))
        checkpoint = torch.load(opt.resume)
        print(f"Loaded state: {checkpoint.keys()}")  # 打印加载的键以调试
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])  # 使用 'model' 作为键名
        else:
            print("Key 'model' not found in checkpoint")
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print("Key 'optimizer' not found in checkpoint")
        opt.start_epoch = checkpoint.get('epoch', 1) + 1  # 设置为检查点中的epoch + 1
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(opt.resume, checkpoint.get('epoch', 'unknown')))
    else:
        print("=> no checkpoint found at '{}'".format(opt.resume))
        opt.start_epoch = 1


def train(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    global_losses = AverageMeter()
    dense_losses = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = torch.cat([images[0], images[1]], dim=0)
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        # compute loss
        dense_features, global_features = model(images)
        global_features_split = torch.split(global_features, [bsz, bsz], dim=0)
        dense_features_split = torch.split(dense_features, [bsz, bsz], dim=0)
        # print(f'Global features shape:{global_features.shape}')
        global_features = torch.cat([f.unsqueeze(1) for f in global_features_split], dim=1)
        dense_features = torch.cat([f.unsqueeze(1) for f in dense_features_split], dim=1)
        # print(f'Global features shape:{global_features.shape}')
        # print(f'Dense features shape: {dense_features.shape}')
        
        if opt.loss_type == 'combined':
            combined_loss, global_loss, dense_loss = criterion(global_features, dense_features)
            loss = combined_loss
        elif opt.loss_type == 'global':
            global_loss = criterion(global_features)
            loss = global_loss
            dense_loss = torch.tensor(0.0).cuda()  # 初始化为零
        elif opt.loss_type == 'dense':
            dense_loss = criterion(dense_features)
            loss = dense_loss
            global_loss = torch.tensor(0.0).cuda()  # 初始化为零


        # update metric
        losses.update(loss.item(), bsz)
        global_losses.update(global_loss.item(), bsz)
        dense_losses.update(dense_loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print(f'Train: [{epoch}][{idx + 1}/{len(train_loader)}]\t'
                  f'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  f'Global Loss {global_losses.val:.3f} ({global_losses.avg:.3f})\t'
                  f'Dense Loss {dense_losses.val:.3f} ({dense_losses.avg:.3f})\t'
                  f'Total Loss {losses.val:.3f} ({losses.avg:.3f})')
            sys.stdout.flush()

    return losses.avg, global_losses.avg, dense_losses.avg


def main():
    opt = parse_option()

    # build data loader
    train_loader = set_loader(opt)

    # build model and criterion
    model, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    # load checkpoint if provided
    load_checkpoint(opt, model, optimizer)

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # training routine
    for epoch in range(opt.start_epoch, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        total_loss, global_loss, dense_loss = train(train_loader, model, criterion, optimizer, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # tensorboard logger
        logger.log_value('loss', total_loss, epoch)
        logger.log_value('global_loss', global_loss, epoch)
        logger.log_value('dense_loss', dense_loss, epoch)
        logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pt'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)

    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pt')
    save_model(model, optimizer, opt, opt.epochs, save_file)


if __name__ == '__main__':
    main()
