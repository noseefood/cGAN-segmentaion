from model.Discriminator import Discriminator
from torchinfo import summary
import torch
import torch.nn as nn

discriminator = Discriminator()
print(discriminator)
# summary(discriminator, (8, 1, 480, 480)) # 1：batch_size 3:图片的通道数 224: 图片的高宽