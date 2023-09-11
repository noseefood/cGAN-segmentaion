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

from util.read_data import SegmentationDataset

import monai

from model.Generator import Generator 
from model.Discriminator import Discriminator

from sklearn.model_selection import train_test_split




def train_loops(args, dataloader, generator, discriminator, optim_G, optim_D, loss_adv, loss_seg, device):
    writer = SummaryWriter() 
    batch_num = 0 

    for epoch in range(args.epoch):

        for i_batch, sample_batched in enumerate(dataloader):  # i_batch: steps
            
            

            batch_num += 1 
            
            # load data
            img, mask = sample_batched['image'], sample_batched['mask']
            valid = Variable(torch.cuda.FloatTensor(mask.size(0), 1).fill_(1.0), requires_grad=False)  # for discriminator 1为真   
            fake = Variable(torch.cuda.FloatTensor(img.size(0), 1).fill_(0.0), requires_grad=False) # for discriminator 0为假

            valid = valid.to(device) 
            fake = fake.to(device) 

            mask = mask.to(device).float()
            img = img.to(device) 

            generator.train()  # recover to train mode(because of eval in validation)
            discriminator.train()  # recover to train mode

            # -----------------
            #  Train Generator
            # -----------------
            optim_G.zero_grad()

            g_output = generator(img)  

            # Loss measures generator's ability to fool the discriminator
            loss_adv_ = loss_adv(discriminator(g_output), valid)

            # Loss measures generator's ability to generate seg mask
            loss_seg_ = loss_seg(g_output, mask) # 
            g_loss = args.lambda_adv * loss_adv_  + args.lambda_seg * loss_seg_  

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
                % (epoch, args.epoch, i_batch, len(dataloader), d_loss.item(), g_loss.item())
            )


            # tensorboard log
            writer.add_scalar('D_loss', d_loss.item(), epoch * len(dataloader) + i_batch)
            writer.add_scalar('G_loss', g_loss.item(), epoch * len(dataloader) + i_batch)

            if batch_num % 150 == 0:
                img_grid = torchvision.utils.make_grid(img, nrow=3, padding=2, normalize=False, range=None, scale_each=False, pad_value=0)
                writer.add_images('input', img_grid, epoch * len(dataloader) + i_batch, dataformats='CHW')
                mask_grid = torchvision.utils.make_grid(mask, nrow=3, padding=2, normalize=False, range=None, scale_each=False, pad_value=0)
                writer.add_images('mask', mask_grid, epoch * len(dataloader) + i_batch, dataformats='CHW')
                g_output_grid = torchvision.utils.make_grid(g_output, nrow=3, padding=2, normalize=False, range=None, scale_each=False, pad_value=0)
                writer.add_images('output', g_output_grid, epoch * len(dataloader) + i_batch, dataformats='CHW')


            # current model save
            if batch_num % (args.save_batch) == 0:

                torch.save(generator.state_dict(), './save_model/save_G_update/generator_'+ str(batch_num) +'.pth')
                torch.save(discriminator.state_dict(), './save_model/save_D_update/discriminator_'+ str(batch_num) +'.pth')
                print("saved current metric model in ", batch_num)

            # validation of generator
            if batch_num % (args.val_batch) == 0:
                generator.eval()
                discriminator.eval()



        # final model save
        torch.save(generator.state_dict(), './save_model/save_G_update/final_generator.pth')
        torch.save(discriminator.state_dict(), './save_model/save_D_update/final_discriminator.pth')





parser = argparse.ArgumentParser()
parser.add_argument('--image_dir', type=str, default='./data/imgs', help='input RGB or Gray image path')
parser.add_argument('--mask_dir', type=str, default='./data/masks', help='input mask path')
parser.add_argument('--lrG', type=float, default='1e-4', help='learning rate')
parser.add_argument('--lrD', type=float, default='5e-5', help='learning rate')
parser.add_argument('--RMSprop', type=bool, default='False', help='RMSprop or Adam')
parser.add_argument('--batch_size', type=int, default='8', help='batch_size in training')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--epoch", type=int, default=500, help="epoch in training")

parser.add_argument("--val_batch", type=int, default=500, help="Every val_batch, do validation")
parser.add_argument("--save_batch", type=int, default=500, help="Every val_batch, do saving model")

parser.add_argument("--lambda_adv", type=float, default=3e-1, help="adversarial loss weight")
parser.add_argument("--lambda_seg", type=float, default=2e-1, help="segmentation loss weight")

args = parser.parse_args()
print('args', args)

os.makedirs('./save_model/save_G_update', exist_ok=True)
os.makedirs('./save_model/save_D_update', exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = SegmentationDataset(args.image_dir, args.mask_dir) 


dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

generator = Generator().to(device)   # input channel must be 1
discriminator = Discriminator().to(device) 

# define optimizer
optim_G = torch.optim.Adam(generator.parameters(), lr=args.lrG, betas=(args.b1, args.b2))
optim_D = torch.optim.Adam(discriminator.parameters(), lr=args.lrD, betas=(args.b1, args.b2))
if args.RMSprop:
    optim_D = torch.optim.RMSprop(discriminator.parameters(), lr = args.lrD)
    optim_G = torch.optim.RMSprop(generator.parameters(), lr = args.lrG)

# define loss
loss_adv = torch.nn.BCELoss().to(device) # 二分类交叉熵 特别针对于GAN adverserial loss
# loss_seg = torch.nn.MSELoss().cuda() # 基本的分割loss
loss_seg = monai.losses.DiceLoss(sigmoid=True).to(device)   # DICE loss, sigmoid参数会让输出的值最后经过sigmoid函数,(input,target)
# loss_seg = torch.nn.BCEWithLogitsLoss().cuda()


# start training loop
train_loops(args, dataloader, generator, discriminator, optim_G, optim_D, loss_adv, loss_seg, device=device)