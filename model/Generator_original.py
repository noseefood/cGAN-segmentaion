import torch
from model.Attention import ChannelAttentionModule, SpatialAttentionModule

class Generator(torch.nn.Module):

    # Unet的skip connections + spatial attentio + channel attention

    def __init__(self, in_features=1, out_features=1, init_features=32):
        super(Generator, self).__init__()
        features = init_features
        self.encode_layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_features, out_channels=features, kernel_size=3, padding=1, stride=1),  # (512-3+2)/1+1=512 
            torch.nn.BatchNorm2d(num_features=features),
            torch.nn.ReLU(),
        )

        # spatial attention: 就是对于所有的通道，在二维平面上，对H x W尺寸的特征图学习到一个权重，对每个像素都会学习到一个权重。你可以想象成一个像素是C维的一个向量，深度是C，在C个维度上，权重都是一样的，但是在平面上，权重不一样
        # Channel Attention：就是对每个C（通道），在channel维度上，学习到不同的权重，平面维度上权重相同。所以基于通道域的注意力通常是对一个通道内的信息直接全局平均池化，而忽略每一个通道内的局部信息

        self.ca = ChannelAttentionModule(init_features)
        self.sa = SpatialAttentionModule() # 

        self.encode_layer1_half = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            stride=1),
            torch.nn.BatchNorm2d(num_features=features),
            torch.nn.ReLU())

        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.encode_layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=features, out_channels=features * 2, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(num_features=features * 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=features * 2, out_channels=features * 2, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(num_features=features * 2),
            torch.nn.ReLU()
        )
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.encode_layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=features * 2, out_channels=features * 4, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(num_features=features * 4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=features * 4, out_channels=features * 4, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(num_features=features * 4),
            torch.nn.ReLU()
        )
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.encode_layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=features * 4, out_channels=features * 8, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(num_features=features * 8),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=features * 8, out_channels=features * 8, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(num_features=features * 8),
            torch.nn.ReLU(),
        )
        self.pool4 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.encode_decode_layer = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=features * 8, out_channels=features * 16, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(num_features=features * 16),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=features * 16, out_channels=features * 16, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(num_features=features * 16),
            torch.nn.ReLU()
        )
        self.upconv4 = torch.nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decode_layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=features * 16, out_channels=features * 8, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(num_features=features * 8),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=features * 8, out_channels=features * 8, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(num_features=features * 8),
            torch.nn.ReLU(),
        )
        self.upconv3 = torch.nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decode_layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=features * 8, out_channels=features * 4, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(num_features=features * 4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=features * 4, out_channels=features * 4, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(num_features=features * 4),
            torch.nn.ReLU()
        )
        self.upconv2 = torch.nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decode_layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=features * 4, out_channels=features * 2, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(num_features=features * 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=features * 2, out_channels=features * 2, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(num_features=features * 2),
            torch.nn.ReLU()
        )
        self.upconv1 = torch.nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decode_layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=features * 2, out_channels=features, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(num_features=features),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(num_features=features),
            torch.nn.ReLU()
        )
        self.out_layer = torch.nn.Sequential(
            # torch.nn.Conv2d(in_channels=features, out_channels=out_features, kernel_size=1, padding=0, stride=1),
            torch.nn.Conv2d(in_channels=features, out_channels=out_features, kernel_size=1, padding=0, stride=1),
        )

    def forward(self, x):
        # cat结构类似UNet
        enc1 = self.encode_layer1(x)
        # print('enc1.shape', enc1.shape)
        ca = self.ca(enc1) * enc1
        # print('ca.shape', ca.shape)
        sa = self.sa(ca) * ca
        # print('sa.shape', sa.shape)

        enc1_half = self.encode_layer1_half(sa)
        enc2 = self.encode_layer2(self.pool1(enc1_half))
        enc3 = self.encode_layer3(self.pool2(enc2))
        enc4 = self.encode_layer4(self.pool3(enc3))
        # print('enc4.shape', enc4.shape)

        bottleneck = self.encode_decode_layer(self.pool4(enc4))
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decode_layer4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decode_layer3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decode_layer2(dec2)

        dec1 = self.upconv1(dec2)

        dec1 = torch.cat((dec1, enc1), dim=1)  # 其实是类似UNet的结构

        dec1 = self.decode_layer1(dec1)

        out = self.out_layer(dec1)

        return out
