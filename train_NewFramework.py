#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   train_final.py
@Time    :   2023/10/04 17:55:38
@Author  :   Xuesong Li
@Version :   1.0
@Contact :   xuesosng.li@tum.de
'''

import argparse
import os
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from util.read_data import SegmentationDataset

from model.Generator import Generator 
from model.Discriminator import Discriminator

import numpy as np
import monai

torch.manual_seed(777)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_loops(args, dataset, split_ratio , generator, discriminator, 
                optim_G, optim_D, loss_adv, loss_seg, metric_val):
    # split train and val dataset
    length =  dataset.num_of_samples()
    train_size = int(0.8 * length) 
    train_set, validate_set = torch.utils.data.random_split(dataset,[train_size,(length-train_size)])

    dataloader_train = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, pin_memory=torch.cuda.is_available())
    dataloader_val = DataLoader(validate_set, batch_size=args.batch_size, shuffle=False, pin_memory=torch.cuda.is_available())

    # define tensorboard writer
    writer = SummaryWriter() 
    batch_num = 0 
    
    for epoch in range(args.epoch):
        for i_batch, sample_batched in enumerate(dataloader_train):  # i_batch: steps
            batch_num += 1 
            # load data
            img, mask = sample_batched['image'], sample_batched['mask']
            valid = Variable(torch.cuda.FloatTensor(mask.size(0), 1).fill_(1.0), requires_grad=False)  # for discriminator 1为真   
            fake = Variable(torch.cuda.FloatTensor(img.size(0), 1).fill_(0.0), requires_grad=False) # for discriminator 0为假
            
            mask = mask.to(device).float()
            img = img.to(device) 

            

parser = argparse.ArgumentParser()
parser.add_argument('--image_dir', type=str, default='C:\Research\projects\Learning\dataset\data_training\Data_Pork/imgs', help='input RGB or Gray image path')
parser.add_argument('--mask_dir', type=str, default='C:\Research\projects\Learning\dataset\data_training\Data_Pork/masks', help='input mask path')
parser.add_argument('--split_ratio', type=float, default='0.8', help='train and val split ratio')

parser.add_argument('--lrG', type=float, default='6e-4', help='learning rate')
parser.add_argument('--lrD', type=float, default='1e-4', help='learning rate') # 
parser.add_argument('--optimizer', type=str, default='Adam', help='RMSprop or Adam')
parser.add_argument('--batch_size', type=int, default='8', help='batch_size in training')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--epoch", type=int, default=500, help="epoch in training")

parser.add_argument("--val_batch", type=int, default=200, help="Every val_batch, do validation")
parser.add_argument("--save_batch", type=int, default=500, help="Every val_batch, do saving model")

args = parser.parse_args()
print('args', args)

os.makedirs('./save_model/save_G_Exp', exist_ok=True)

dataset = SegmentationDataset(args.image_dir, args.mask_dir) 

generator = Generator().to(device)   # input channel must be 1
discriminator = Discriminator().to(device) 

# define optimizer
if args.optimizer == "RMSprop":
    optim_D = torch.optim.RMSprop(discriminator.parameters(), lr = args.lrD)
    optim_G = torch.optim.RMSprop(generator.parameters(), lr = args.lrG)
elif args.optimizer == "Adam": 
    optim_G = torch.optim.Adam(generator.parameters(), lr=args.lrG, betas=(args.b1, args.b2))
    optim_D = torch.optim.Adam(discriminator.parameters(), lr=args.lrD, betas=(args.b1, args.b2))
elif args.optimizer == "SGD":
    optim_G = torch.optim.SGD(generator.parameters(), lr=args.lrG, momentum=0.9)
    optim_D = torch.optim.SGD(discriminator.parameters(), lr=args.lrD, momentum=0.9)

# define loss
loss_adv = torch.nn.BCELoss().to(device) # GAN adverserial loss
loss_seg = torch.nn.MSELoss().to(device) # 基本的分割loss
metric_val = monai.metrics.DiceHelper(sigmoid=True) # DICE score for validation of generator 最终输出的时候也应该经过sigmoid函数!!!!!!!!!!!!!!!!!!!!!!
# loss_seg = monai.losses.DiceLoss(sigmoid=True).to(device)   # DICE loss, sigmoid参数会让输出的值最后经过sigmoid函数,(input,target)
# loss_seg = torch.nn.BCEWithLogitsLoss().cuda() # BECWithLogitsLoss即是把最后的sigmoid和BCELoss合成一步，效果是一样的
# loss_seg =  monai.losses.FocalLoss().to(device) # FocalLoss is an extension of BCEWithLogitsLoss, so sigmoid is not needed.

train_loops(args, dataset, generator, discriminator, optim_G, optim_D, loss_adv, loss_seg, metric_val, device=device)