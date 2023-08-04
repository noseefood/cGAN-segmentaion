import torch
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import torchvision


def train(args, dataloader, generator, discriminator, optim_G, optim_D, loss_adv, loss_rec):

    writer = SummaryWriter()  

    for epoch in range(args.epoch):
        for i_batch, sample_batched in enumerate(dataloader):  # i_batch: steps

            # update generator

            img, mask = sample_batched['image'], sample_batched['mask']

            valid = Variable(torch.cuda.FloatTensor(mask.size(0), 1).fill_(1.0), requires_grad=False)  # for discriminator
            fake = Variable(torch.cuda.FloatTensor(img.size(0), 1).fill_(0.0), requires_grad=False) # for discriminator

            valid = valid.cuda()
            fake = fake.cuda()

            mask = mask.cuda()
            img = img.cuda()

            optim_G.zero_grad()

            g_output = generator(img)  # predicted mask

            loss_adv_ = loss_adv(discriminator(g_output), valid)

            mask = mask.float()

            loss_rec_ = loss_rec(g_output, mask)
            g_loss = (loss_adv_ + loss_rec_) / 2

            g_loss.backward()
            optim_G.step()



            # update discriminator

            optim_D.zero_grad()

            # print('discriminator(mask)', discriminator(mask).shape)
            # print('valid', valid.shape)
            real_loss = loss_adv(discriminator(mask), valid)
            fake_loss = loss_adv(discriminator(g_output.detach()), fake)

            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward(retain_graph=True)
            optim_D.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, args.epoch, i_batch, len(dataloader), d_loss.item(), g_loss.item())
            )


            writer.add_scalar('D_loss', d_loss.item(), epoch * len(dataloader) + i_batch)
            writer.add_scalar('G_loss', g_loss.item(), epoch * len(dataloader) + i_batch)
            
            # img_grid = torchvision.utils.make_grid(
            #     tensor,  # 图像数据，B x C x H x W 形式
            #     nrow=8,  # 一行显示 nrow 个
            #     padding=2,  # 图像间距（像素单位）
            #     normalize=False,  # 是否将像素值标准化，默认为 False。通常网络中的图片像素值比较小，要可视化之前需要标准化到0~255。
            #     range=None,  # 截断范围。譬如，若像素值范围在 0~255，传入 (100, 200)，则小于 100 的都会变为 100，大于 200 的都会变为 200。
            #     scale_each=False,  # 是否单张图维度标准化，默认为 False
            #     pad_value=0,  # 子图之间 padding 的像素值
            #     ) 

            if i_batch % 200 == 0:
                img_grid = torchvision.utils.make_grid(img, nrow=3, padding=2, normalize=False, range=None, scale_each=False, pad_value=0)
                writer.add_images('input', img_grid, epoch * len(dataloader) + i_batch, dataformats='CHW')
                mask_grid = torchvision.utils.make_grid(mask, nrow=3, padding=2, normalize=False, range=None, scale_each=False, pad_value=0)
                writer.add_images('mask', mask_grid, epoch * len(dataloader) + i_batch, dataformats='CHW')
                g_output_grid = torchvision.utils.make_grid(g_output, nrow=3, padding=2, normalize=False, range=None, scale_each=False, pad_value=0)
                writer.add_images('output', g_output_grid, epoch * len(dataloader) + i_batch, dataformats='CHW')

            # best model save


        # test?
        generator.eval()


        torch.save(generator.state_dict(), './save_model/save_G/final_generator.pth')
        torch.save(discriminator.state_dict(), './save_model/save_D/final_discriminator.pth')
