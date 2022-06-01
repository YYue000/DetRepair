from imagecorruptions import corrupt
import torchvision
import torch
import numpy as np
from torchvision import transforms

import os
from functools import partial
import argparse

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(filename)s#%(lineno)d:%(message)s')
logger = logging.getLogger('global')


class Corrupt:
    def __init__(self, corruption, severity):
        self.corruption = corruption
        self.severity = severity
    def __call__(self, pic):
        pic = np.array(pic).astype(np.uint8)
        return corrupt(pic, corruption_name=self.corruption, severity=self.severity)


def get_dataloader(args):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    corruption = Corrupt(args.corruption, args.severity)
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        corruption,
        transforms.ToTensor(),
        normalize])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        corruption,
        transforms.ToTensor(),
        normalize])
    val_transform_raw = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize])


    train_dataset = torchvision.datasets.ImageNet(args.data_root, split='train', transform=train_transform)
    val_dataset = torchvision.datasets.ImageNet(args.data_root, split='val', transform=val_transform)
    val_dataset_raw = torchvision.datasets.ImageNet(args.data_root, split='val', transform=val_transform_raw)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    val_dataloader_raw = torch.utils.data.DataLoader(val_dataset_raw, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_dataloader, val_dataloader, val_dataloader_raw

def get_correct_k(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            #res.append(correct_k.mul_(100.0 / batch_size))
            res.append(correct_k)
        return res

def validate(dataloader, model, args):
    model.eval()

    acc1, acc5 = 0,0
    N = 0
    print_freq = args.val_print_freq
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            img, label = data
            if args.gpu is not None:
                img = img.to(device=args.gpu)
            pred = model(img).cpu()
            c1, c5 = get_correct_k(pred, label, (1,5))
            acc1 += c1
            acc5 += c5
            N += len(label)
            if print_freq > 0 and i%print_freq == 0:
                logger.info(f'iter {i}: acc1{c1/len(label)} acc5{c5/len(label)}')
    acc1 /= N
    acc5/=N
    return acc1, acc5

def calibration(train_dataloader, val_dataloader, val_dataset_raw, model, args):
    model.train()
    if not os.path.exists(args.ckpt_save_path):
        os.makedirs(args.ckpt_save_path)
    with torch.no_grad():
        for i, data in enumerate(train_dataloader):
            img, label = data
            if args.gpu is not None:
                img = img.to(device=args.gpu)

            model(img)
            if i%args.train_print_freq==0:
                acc1_raw, _ = validate(val_dataloader_raw, model, args)
                acc1, _ = validate(val_dataloader, model, args)
                logger.info(f'iter {i}: acc1 {acc1} {acc1_raw}')
            if i%args.ckpt_freq==0:
                torch.save(model.state_dict(),f'{args.ckpt_save_path}/ckpt_iter{i}.pth')
                model.train()
        torch.save(model.state_dict(),f'checkpoints/ckpt.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--corruption", type=str)
    parser.add_argument("--severity", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--data_root", default="/yueyuxin/data/imagenet")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--ckpt_freq", type=int, default=100)
    parser.add_argument("--train_print_freq", type=int, default=20)
    parser.add_argument("--val_print_freq", type=int, default=-1)
    parser.add_argument("--ckpt_save_path", type=str, default="checkpoints")
    args = parser.parse_args()
    
    train_dataloader, val_dataloader, val_dataloader_raw = get_dataloader(args)
    logger.info(f'dataloaders {len(train_dataloader)} {len(val_dataloader)}')
    #model = torchvision.models.alexnet(pretrained=True)
    model = torchvision.models.resnet50(pretrained=True)
    if args.gpu is not None:
        model.cuda().to(device=args.gpu)
    
    acc_raw = validate(val_dataloader_raw, model, args)
    acc = validate(val_dataloader, model, args)
    logger.info(f'accuracy {acc} {acc_raw}')
    logger.info('-'*20)

    calibration(train_dataloader, val_dataloader, val_dataloader_raw, model, args)
