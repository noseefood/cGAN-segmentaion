import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(1, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image(final dense layer)
        # ds_size = 480 // 2 ** 4
        ds_size = 512 // 2 ** 4

        # Output layers Wasserstein GAN improvement(Discriminator output no sigmoid!)
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid()) # 128, 30, 30
        # self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1)) # 128, 30, 30

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1) #

        validity = self.adv_layer(out)

        # print("validity", validity.shape) # torch.Size([8, 1])

        return validity
