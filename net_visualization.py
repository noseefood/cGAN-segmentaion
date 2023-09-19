from model.Discriminator import Discriminator
from model.Generator import Generator
from torchinfo import summary
import torch
import torch.nn as nn
from torchview import draw_graph

# discriminator = Discriminator()
# print(discriminator)
# summary(discriminator, (8, 1, 480, 480)) # 1：batch_size 3:图片的通道数 224: 图片的高宽

# generator = Generator()
# # print(generator)
# summary(generator, (8, 1, 512, 512))


model = Generator()
batch_size = 8
# device='meta' -> no memory is consumed for visualization
model_graph = draw_graph(model, input_size=(batch_size,1,512,512), device='cuda')
model_graph.visual_graph