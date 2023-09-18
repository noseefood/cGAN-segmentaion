#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   train_final.py
@Time    :   2023/09/10 17:55:38
@Author  :   Xuesong Li
@Version :   1.0
@Contact :   xuesosng.li@tum.de
'''
'''
reference: 
    https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/dcgan/dcgan.py
    https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/esrgan/esrgan.py  (how to combine two loss parts for generator)
'''
import argparse
import os
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
# from torchmetrics.functional import dice_score

from util.read_data import SegmentationDataset

import monai

from model.Generator import Generator 
from model.Discriminator import Discriminator
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    ScaleIntensity,
    Resize,
    AddChannel,
)
import numpy as np
from sklearn.model_selection import train_test_split

# gamma_max = 0.05
torch.manual_seed(777)
if_adersial = False
# tf = Compose([AsDiscrete(threshold=0.5)])

def train_loops(args, dataloader_train, dataloader_val, generator, discriminator, optim_G, optim_D, loss_adv, loss_seg, metric_val, device):
    writer = SummaryWriter() 
    batch_num = 0 
    # gamma = 0
    for epoch in range(args.epoch):

        for i_batch, sample_batched in enumerate(dataloader_train):  # i_batch: steps
            

            batch_num += 1 
            
            # load data
            img, mask = sample_batched['image'], sample_batched['mask']
            valid = Variable(torch.cuda.FloatTensor(mask.size(0), 1).fill_(1.0), requires_grad=False)  # for discriminator 1为真   
            fake = Variable(torch.cuda.FloatTensor(img.size(0), 1).fill_(0.0), requires_grad=False) # for discriminator 0为假

            # valid = valid.to(device) 
            # fake = fake.to(device) 

            mask = mask.to(device).float()
            img = img.to(device) 
            # mask = mask.float()

            generator.train()  # recover to train mode(because of eval in validation)
            discriminator.train()  # recover to train mode

            # -----------------
            #  Train Generator
            # -----------------
            optim_G.zero_grad()

            g_output = generator(img) 
            # g_output = tf(g_output)  #

            # Loss measures generator's ability to fool the discriminator
            loss_adv_ = loss_adv(discriminator(g_output), valid)

            # Loss measures generator's ability to generate seg mask
            loss_seg_ = loss_seg(g_output, mask) # 
            # g_loss = args.lambda_adv * loss_adv_ * gamma  + args.lambda_seg * loss_seg_  
            g_loss = args.lambda_adv * loss_adv_   + args.lambda_seg * loss_seg_  

            g_loss.backward()
            optim_G.step()

            print("loss_adv_", loss_adv_.item())
            print("loss_seg_", loss_seg_.item())

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optim_D.zero_grad()

            real_loss = loss_adv(discriminator(mask), valid) # 能不能区分出真实的mask 二分类交叉熵 BCELoss
            fake_loss = loss_adv(discriminator(g_output.detach()), fake)  # 能不能区分出虚假的mask 二分类交叉熵 BCELoss
            d_loss = (real_loss + fake_loss) / 2

            # d_loss.backward(retain_graph=True)
            d_loss.backward()
            optim_D.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, args.epoch, i_batch, len(dataloader_train), d_loss.item(), g_loss.item())
            )


            # tensorboard log
            writer.add_scalar('D_loss', d_loss.item(), epoch * len(dataloader_train) + i_batch)
            writer.add_scalar('G_loss', g_loss.item(), epoch * len(dataloader_train) + i_batch)

            if batch_num % 150 == 0:
                img_grid = torchvision.utils.make_grid(img, nrow=3, padding=2, normalize=False, range=None, scale_each=False, pad_value=0)
                writer.add_images('input', img_grid, epoch * len(dataloader_train) + i_batch, dataformats='CHW')
                mask_grid = torchvision.utils.make_grid(mask, nrow=3, padding=2, normalize=False, range=None, scale_each=False, pad_value=0)
                writer.add_images('mask', mask_grid, epoch * len(dataloader_train) + i_batch, dataformats='CHW')
                g_output_grid = torchvision.utils.make_grid(g_output, nrow=3, padding=2, normalize=False, range=None, scale_each=False, pad_value=0)
                writer.add_images('output', g_output_grid, epoch * len(dataloader_train) + i_batch, dataformats='CHW')


            # current model save
            if batch_num % (args.save_batch) == 0:

                torch.save(generator.state_dict(), './save_model/save_G_update/generator_'+ str(batch_num) +'.pth')
                torch.save(discriminator.state_dict(), './save_model/save_D_update/discriminator_'+ str(batch_num) +'.pth')
                print("saved current metric model in ", batch_num)

            # validation of generator
            if batch_num % (args.val_batch) == 0:

                generator.eval()
                # discriminator.eval()
                val_scores = []    
                with torch.no_grad():
                    for i_batch_val, sample_batched in enumerate(dataloader_val):  # i_batch: steps
                        
                        img, mask = sample_batched['image'], sample_batched['mask']
                        mask = mask.to(device).float() # ([8, 1, 512, 512])
                        img = img.to(device) 

                        g_output = generator(img) # ([8, 1, 512, 512])
                        # g_output = tf(g_output)  #
                        dice_cof, _ = metric_val(y_pred=g_output, y=mask)
                        
                        val_scores.append(dice_cof.cpu().numpy())

                print("val_scores", val_scores)
                metric = np.mean(val_scores)
                print("mean dice score: ", metric)

                writer.add_scalar("val_mean_dice", metric, epoch * len(dataloader_train) + i_batch)
                
            # if batch_num % 1000 == 0 and gamma < gamma_max:
            #     gamma += 0.01
            #     print("Current gamma: ", gamma)
            #     writer.add_scalar("Current gamma decay", gamma, epoch * len(dataloader_train) + i_batch)
            


        # final model save
        if epoch == args.epoch - 1:
            print("final model saved")
            torch.save(generator.state_dict(), './save_model/save_G_update/final_generator.pth')
            torch.save(discriminator.state_dict(), './save_model/save_D_update/final_discriminator.pth')


def train_loops_Generaotr(args, dataloader_train, dataloader_val, generator, optim_G, loss_seg, metric_val, device):
    '''
    only train generator, no discriminator, equivalent to commonly segmentation training
    '''
    writer = SummaryWriter() 
    batch_num = 0 

    for epoch in range(args.epoch):

        for i_batch, sample_batched in enumerate(dataloader_train):  # i_batch: steps
            
            batch_num += 1 
            
            # load data
            img, mask = sample_batched['image'], sample_batched['mask']

            mask = mask.to(device).float()
            img = img.to(device) 

            generator.train()  # recover to train mode(because of eval in validation)

            # -----------------
            #  Train Generator
            # -----------------
            optim_G.zero_grad()

            g_output = generator(img) 
            # g_output = tf(g_output)  

            # Loss measures generator's ability to generate seg mask
            loss_seg_ = loss_seg(g_output, mask) # 
            # g_loss = args.lambda_adv * loss_adv_ * gamma  + args.lambda_seg * loss_seg_  
            g_loss = loss_seg_  

            g_loss.backward()
            optim_G.step()

            print("loss_seg_", loss_seg_.item())


            print(
                "[Epoch %d/%d] [Batch %d/%d] [G loss: %f]"
                % (epoch, args.epoch, i_batch, len(dataloader_train), g_loss.item())
            )


            # tensorboard 
            writer.add_scalar('G_loss', g_loss.item(), epoch * len(dataloader_train) + i_batch)

            if batch_num % 150 == 0:
                img_grid = torchvision.utils.make_grid(img, nrow=3, padding=2, normalize=False, range=None, scale_each=False, pad_value=0)
                writer.add_images('input', img_grid, epoch * len(dataloader_train) + i_batch, dataformats='CHW')
                mask_grid = torchvision.utils.make_grid(mask, nrow=3, padding=2, normalize=False, range=None, scale_each=False, pad_value=0)
                writer.add_images('mask', mask_grid, epoch * len(dataloader_train) + i_batch, dataformats='CHW')
                g_output_grid = torchvision.utils.make_grid(g_output, nrow=3, padding=2, normalize=False, range=None, scale_each=False, pad_value=0)
                writer.add_images('output', g_output_grid, epoch * len(dataloader_train) + i_batch, dataformats='CHW')


            # current model save
            if batch_num % (args.save_batch) == 0:

                torch.save(generator.state_dict(), './save_model/save_G_Only/generator_'+ str(batch_num) +'.pth')
                print("saved current metric model in ", batch_num)

            # validation of generator
            if batch_num % (args.val_batch) == 0:

                generator.eval()
                val_scores = []    
                with torch.no_grad():
                    for i_batch_val, sample_batched in enumerate(dataloader_val):  # i_batch: steps
                        
                        img, mask = sample_batched['image'], sample_batched['mask']
                        mask = mask.to(device).float() # ([8, 1, 512, 512])
                        img = img.to(device) 

                        g_output = generator(img) # ([8, 1, 512, 512])
                        # g_output = tf(g_output)  #
                        dice_cof, _ = metric_val(y_pred=g_output, y=mask)
                        
                        val_scores.append(dice_cof.cpu().numpy())

                print("val_scores", val_scores)
                metric = np.mean(val_scores)
                print("mean dice score: ", metric)

                writer.add_scalar("val_mean_dice", metric, epoch * len(dataloader_train) + i_batch)
                    


        # final model save
        if epoch == args.epoch - 1:
            print("final model saved")
            torch.save(generator.state_dict(), './save_model/save_G_Only/final_generator.pth')


parser = argparse.ArgumentParser()
parser.add_argument('--image_dir', type=str, default='./data/imgs', help='input RGB or Gray image path')
parser.add_argument('--mask_dir', type=str, default='./data/masks', help='input mask path')
parser.add_argument('--lrG', type=float, default='2e-4', help='learning rate')
parser.add_argument('--lrD', type=float, default='5e-5', help='learning rate')
parser.add_argument('--optimizer', type=str, default='SGD', help='RMSprop or Adam')
parser.add_argument('--batch_size', type=int, default='8', help='batch_size in training')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--epoch", type=int, default=500, help="epoch in training")

parser.add_argument("--val_batch", type=int, default=200, help="Every val_batch, do validation")
parser.add_argument("--save_batch", type=int, default=500, help="Every val_batch, do saving model")

parser.add_argument("--lambda_adv", type=float, default=0.2, help="adversarial loss weight")
parser.add_argument("--lambda_seg", type=float, default=1, help="segmentation loss weight")

args = parser.parse_args()
print('args', args)

os.makedirs('./save_model/save_G_update', exist_ok=True)
os.makedirs('./save_model/save_D_update', exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# define dataset
dataset = SegmentationDataset(args.image_dir, args.mask_dir) 
length =  dataset.num_of_samples()
train_size, validate_size=int(0.8 * length),int(0.2 * length)
train_set, validate_set=torch.utils.data.random_split(dataset,[train_size,validate_size])

dataloader_train = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, pin_memory=torch.cuda.is_available())
dataloader_val = DataLoader(validate_set, batch_size=args.batch_size, shuffle=False, pin_memory=torch.cuda.is_available())
# dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, pin_memory=torch.cuda.is_available())

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
# loss_seg = torch.nn.MSELoss().to(device) # 基本的分割loss
metric_val = monai.metrics.DiceHelper(sigmoid=True) # DICE score for validation of generator 最终输出的时候也应该经过sigmoid函数!!!!!!!!!!!!!!!!!!!!!!
loss_seg = monai.losses.DiceLoss(sigmoid=True).to(device)   # DICE loss, sigmoid参数会让输出的值最后经过sigmoid函数,(input,target)
# loss_seg = torch.nn.BCEWithLogitsLoss().cuda() # BECWithLogitsLoss即是把最后的sigmoid和BCELoss合成一步，效果是一样的
# loss_seg =  monai.losses.FocalLoss().to(device) # FocalLoss is an extension of BCEWithLogitsLoss, so sigmoid is not needed.


# start training loop
if if_adersial:
    train_loops(args, dataloader_train, dataloader_val, generator, discriminator, optim_G, optim_D, loss_adv, loss_seg, metric_val, device=device)
else:
    train_loops_Generaotr(args, dataloader_train, dataloader_val, generator, optim_G, loss_seg, metric_val, device=device)