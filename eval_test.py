from model.Generator_original import Generator as Generator_original # original UAGAN
from model.Generator import Generator # New structure

import os
import sys
from glob import glob
import cv2
import time
import torch
from PIL import Image
import numpy as np
import monai
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    ScaleIntensity,
    Resize,
    AddChannel,
)
from monai.networks.nets import UNet, AttentionUnet

Video_recording = True


# Dice similarity function
def dice(pred_mask, gt_mask, k = 1):
    intersection = np.sum(pred_mask[gt_mask==k]) * 2.0
    dice = intersection / (np.sum(pred_mask) + np.sum(gt_mask))
    return dice

def calculate_iou(gt_mask, pred_mask, cls=255):
    '''cls 为二值化的max值,比如255'''
    '''miou可解释为平均交并比,即在每个类别上计算IoU值,然后求平均值,在这里等价于iou'''
    pred_mask = (pred_mask == cls) * 1
    gt_mask = (gt_mask == cls) * 1

    overlap = pred_mask * gt_mask  # Logical AND
    union = (pred_mask + gt_mask) > 0  # Logical OR
    iou = overlap.sum() / float(union.sum())

    return iou

class NetworkInference_GANVer2():
    '''
    updated generator, princile is the same as UAGAN(similar to Unet+double attention)
    '''
    def __init__(self, mode = "pork"):

        dir_checkpoint_GAN = '/home/xuesong/generator_9000.pth'
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.generator = Generator().to(self.device)  # input channel = 1
        self.generator.load_state_dict(torch.load(dir_checkpoint_GAN))
        self.generator.eval() # eval mode

        self.train_imtrans = Compose( # 预处理
            [   
                AddChannel(),  # 增加维度
                Resize((512, 512)), # 必须要加入这个，否则会报错，这里相当于直接拉伸，跟training保持一致
                ScaleIntensity(), # 归一化 0-255 -> 0-1
            ]
        )
        # self.post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])  # 这里不需要sigmoid，因为网络中已经有了sigmoid
        self.tf = Compose([Resize((657, 671)), AsDiscrete(threshold=0.5)])  # 先拉伸到原来的大小, 别忘了asdiscrete二值化

    def inference(self, img, tf = None):
        
        with torch.no_grad():
            
            # TODO:检测输入图片的通道数，如果是3通道，需要转换为1通道

            img = self.train_imtrans(img) # compose会自动返回tensor torch.Size([1, 512, 512])

            img = img.to(self.device) # torch.Size([1, 512, 512])   HWC to CHW：img_trans = img_nd.transpose((2, 0, 1))
            img = img.unsqueeze(0) # torch.Size([1, 1, 512, 512]) unsqueeze扩增维度

            output = self.generator(img)

            probs = output.squeeze(0) # squeeze压缩维度 torch.Size([1, 512, 512])

            if tf is not None:
                self.tf = tf

            probs = self.tf(probs.cpu()) # 重新拉伸到原来的大小
            full_mask = probs.squeeze().cpu().numpy() # return in cpu  # 
            # cv2.imshow("full_mask", full_mask)
            
            return full_mask

class NetworkInference_GanPlusAttUnet():
    '''Attention Unet training using combined loss between adversial loss and segmentation loss'''
    def __init__(self):
        monai.config.print_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

        # self.model = AttentionUnet(
        #     spatial_dims=2,
        #     in_channels=1,
        #     out_channels=1,
        #     channels=(16, 32, 64, 128, 256),
        #     strides=(2, 2, 2, 2),  
        #     kernel_size=3,
        # ).to(self.device)   
        # self.model.load_state_dict(torch.load("/home/xuesong/CAMP/segment/cGAN-segmentaion/save_model/trained_model/04.09.AttUnet/save_G_update/generator_36500.pth"))
        self.model = monai.networks.nets.AttentionUnet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            channels=(32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2),  
            kernel_size=3,
        ).to(self.device)  
        self.model.load_state_dict(torch.load("/home/xuesong/CAMP/segment/cGAN-segmentaion/save_model/trained_model/09.09.AttUnet/save_G_update/generator_39000.pth"))

        self.model.eval()

        self.train_imtrans = Compose( # 输入模型的图片的预处理
            [   
                AddChannel(),  # 增加通道维度
                Resize((512, 512)), # 必须要加入这个，否则会报错，这里相当于直接拉伸，跟training保持一致
                ScaleIntensity(), # 其实就是归一化
            ]
        )
        
        # self.post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
        self.tf = Compose([Resize((657, 671)), AsDiscrete(threshold=0.3)])

    def inference(self, img, tf = None):
        with torch.no_grad():
            img = self.train_imtrans(img) # compose会自动返回tensor torch.Size([1, 512, 512])

            img = img.to(self.device) # torch.Size([1, 512, 512])   HWC to CHW：img_trans = img_nd.transpose((2, 0, 1))
            img = img.unsqueeze(0) # torch.Size([1, 1, 512, 512]) unsqueeze扩增维度
            
            # # 因为这里没有使用dataloader读取，所以不需要转置，输出可以直接与原图对比
            # img = img.transpose(-1,-2) # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 注意这里与inference.py不同，这里不需要转置，以为读取图片的方式不一样！

            output = self.model(img)
            # result = self.post_trans(output) # torch.Size([1, 1, 512, 512])
            result = output

            probs = result.squeeze(0) # squeeze压缩维度 torch.Size([1, 512, 512])
            
            if tf is not None:
                self.tf = tf

            probs = self.tf(probs.cpu()) # 重新拉伸到原来的大小

            full_mask = probs.squeeze().cpu().numpy() # return in cpu  # 
            
            # cv2.imshow("full_mask", full_mask)
            # cv2.waitKey(0)

            return full_mask

class NetworkInference_Unet():
    def __init__(self, mode = "pork", method = "Unet"):

        monai.config.print_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        \
        if method == "AttentionUnet":
            self.model = AttentionUnet(
                spatial_dims=2,
                in_channels=1,
                out_channels=1,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),  
                kernel_size=3,
            ).to(self.device)   
            self.model.load_state_dict(torch.load("/home/xuesong/CAMP/segment/selfMonaiSegment/best_metric_model_AttentionUnet.pth"))
            # self.model.load_state_dict(torch.load("/home/xuesong/CAMP/segment/cGAN-segmentaion/save_model/trained_model/04.09.AttUnet/save_G_update/generator_36500.pth"))

        elif method == "Unet":
            self.model = UNet(
                spatial_dims=2,
                in_channels=1,
                out_channels=1,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
                num_res_units=2,
            ).to(self.device)
            self.model.load_state_dict(torch.load("/home/xuesong/CAMP/segment/selfMonaiSegment/best_metric_model_Unet.pth"))

        self.model.eval()

        self.train_imtrans = Compose( # 输入模型的图片的预处理
            [   
                AddChannel(),  # 增加通道维度
                Resize((512, 512)), # 必须要加入这个，否则会报错，这里相当于直接拉伸，跟training保持一致
                ScaleIntensity(), # 其实就是归一化
            ]
        )
        
        self.post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
        self.tf = Compose([Resize((657, 671)),])

    def inference(self, img, tf = None):
        with torch.no_grad():
            img = self.train_imtrans(img) # compose会自动返回tensor torch.Size([1, 512, 512])

            img = img.to(self.device) # torch.Size([1, 512, 512])   HWC to CHW：img_trans = img_nd.transpose((2, 0, 1))
            img = img.unsqueeze(0) # torch.Size([1, 1, 512, 512]) unsqueeze扩增维度
            
            # # 因为这里没有使用dataloader读取，所以不需要转置，输出可以直接与原图对比
            # img = img.transpose(-1,-2) # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 注意这里与inference.py不同，这里不需要转置，以为读取图片的方式不一样！

            output = self.model(img)
            result = self.post_trans(output) # torch.Size([1, 1, 512, 512])

            probs = result.squeeze(0) # squeeze压缩维度 torch.Size([1, 512, 512])
            
            if tf is not None:
                self.tf = tf

            probs = self.tf(probs.cpu()) # 重新拉伸到原来的大小

            full_mask = probs.squeeze().cpu().numpy() # return in cpu  # 
            
            # cv2.imshow("full_mask", full_mask)
            # cv2.waitKey(0)

            return full_mask


class NetworkInference_GAN():
    '''
    original generator from UAGAN
    '''
    def __init__(self, mode = "pork"):

        dir_checkpoint_GAN = '/home/xuesong/CAMP/segment/cGAN-segmentaion/data/models/05.08/runs/generator_26500.pth'
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.generator = Generator_original().to(self.device)  # input channel = 1
        # self.generator.load_state_dict(torch.load(dir_checkpoint + "UAGAN_generator.pth"))
        self.generator.load_state_dict(torch.load(dir_checkpoint_GAN))
        self.generator.eval() # eval mode

        self.train_imtrans = Compose( # 预处理
            [   
                AddChannel(),  # 增加维度
                Resize((480, 480)), # 必须要加入这个，否则会报错，这里相当于直接拉伸，跟training保持一致(注意这里之前训练的时候是resize到480,后来都改成512了)
                ScaleIntensity(), # 归一化 0-255 -> 0-1
            ]
        )
        # self.post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])  # 这里不需要sigmoid，因为网络中已经有了sigmoid
        self.tf = Compose([Resize((657, 671)), AsDiscrete(threshold=0.5)])  # 先拉伸到原来的大小, 别忘了asdiscrete二值化

    def inference(self, img, tf = None):
        
        with torch.no_grad():
            
            # TODO:检测输入图片的通道数，如果是3通道，需要转换为1通道

            img = self.train_imtrans(img) # compose会自动返回tensor torch.Size([1, 512, 512])

            img = img.to(self.device) # torch.Size([1, 512, 512])   HWC to CHW：img_trans = img_nd.transpose((2, 0, 1))
            img = img.unsqueeze(0) # torch.Size([1, 1, 512, 512]) unsqueeze扩增维度
            # # 因为这里没有使用dataloader读取，所以不需要转置，输出可以直接与原图对比
            # img = img.transpose(-1,-2) # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 注意这里与inference.py不同，这里不需要转置，以为读取图片的方式不一样！

            output = self.generator(img)

            probs = output.squeeze(0) # squeeze压缩维度 torch.Size([1, 512, 512])

            if tf is not None:
                self.tf = tf

            probs = self.tf(probs.cpu()) # 重新拉伸到原来的大小
            full_mask = probs.squeeze().cpu().numpy() # return in cpu  # 
            # cv2.imshow("full_mask", full_mask)
            
            return full_mask
        




class Evaluation():
    
    def __init__(self, mode = "1"):

        self.dataPath = '/home/xuesong/CAMP/dataset/datasetTest_080823/test_dataset/' + mode
        # self.net_GAN = NetworkInference_GAN("pork")
        self.net_GAN = NetworkInference_GANVer2("pork")
        self.net_Unet = NetworkInference_Unet("pork", method = "Unet")  # "Unet" or "AttentionUnet" for comparison
        self.net_AttUnet = NetworkInference_Unet("pork", method = "AttentionUnet")  # "Unet" or "AttentionUnet" for comparison
        self.net_GANPlusAttUnet = NetworkInference_GanPlusAttUnet()

        print("dataPath:", self.dataPath)
        self.imgs = glob(self.dataPath + '/imgs' +"/*.png")
        self.masks = glob(self.dataPath + '/masks' +"/*.png") 
        self.imgs.sort()
        self.masks.sort()

        if Video_recording:
            fps = 25
            img_size = (671,657) # 50 mm depth
            fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
            # fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            video_dir_img = '/home/xuesong/CAMP/segment/cGAN-segmentaion/results/img_' + mode + '.avi'
            video_dir_mask = '/home/xuesong/CAMP/segment/cGAN-segmentaion/results/mask_' + mode + '.avi'
            video_dir_GAN = '/home/xuesong/CAMP/segment/cGAN-segmentaion/results/GAN_' + mode + '.avi'
            video_dir_Unet = '/home/xuesong/CAMP/segment/cGAN-segmentaion/results/Unet_' + mode + '.avi'
            video_dir_AttUnet = '/home/xuesong/CAMP/segment/cGAN-segmentaion/results/AttUnet_' + mode + '.avi'
            video_dir_GANPlusAttUnet = '/home/xuesong/CAMP/segment/cGAN-segmentaion/results/GANPlusAttUnet_' + mode + '.avi'

            self.videoWriter_img = cv2.VideoWriter(video_dir_img, fourcc, fps, img_size, isColor=True)
            self.videoWriter_mask = cv2.VideoWriter(video_dir_mask, fourcc, fps, img_size, isColor=False)
            self.videoWriter_GAN = cv2.VideoWriter(video_dir_GAN, fourcc, fps, img_size, isColor=False)
            self.videoWriter_Unet = cv2.VideoWriter(video_dir_Unet, fourcc, fps, img_size, isColor=False)
            self.videoWriter_AttUnet = cv2.VideoWriter(video_dir_AttUnet, fourcc, fps, img_size, isColor=False)
            self.videoWriter_GANPlusAttUnet = cv2.VideoWriter(video_dir_GANPlusAttUnet, fourcc, fps, img_size, isColor=False)

    def start(self):

        assert len(self.imgs) == len(self.masks), \
            f'len(self.imgs) should be equal to len(self.masks), but got {len(self.imgs)} vs {len(self.masks)}'

        print("All frames number:", len(self.imgs))

        dice_list_GAN = []
        dice_list_Unet = []
        dice_list_AttUnet = []
        dice_list_GANPlusAttUnet = []
        iou_list_GAN = []
        iou_list_Unet = []
        iou_list_AttUnet = []
        iou_list_GANPlusAttUnet = []



        for i,(input, target) in enumerate(zip(self.imgs,self.masks)):
            
            img = cv2.imread(input)
            true_mask = cv2.imread(target) 
            true_mask = cv2.cvtColor(true_mask, cv2.COLOR_BGR2GRAY) * 255 # 0-1 -> 0-255

            output_Gan = self.net_GAN.inference(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)) # 0-1
            output_Unet = self.net_Unet.inference(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)) # 0-1
            output_AttUnet = self.net_AttUnet.inference(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)) # 0-1
            output_GANPlusAttUnet = self.net_GANPlusAttUnet.inference(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)) # 0-1

            output_Gan = cv2.normalize(output_Gan, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U) # 0-1 -> 0-255
            output_Unet = cv2.normalize(output_Unet, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U) # 0-1 -> 0-255
            output_AttUnet = cv2.normalize(output_AttUnet, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U) # 0-1 -> 0-255
            output_GANPlusAttUnet = cv2.normalize(output_GANPlusAttUnet, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U) # 0-1 -> 0-255


            # print("Image data type:", output_Gan.dtype)
            # print("Image data type:", output_Unet.dtype)

            cv2.imshow("output_GAN", output_Gan)    
            cv2.imshow("output_Unet", output_Unet)
            cv2.imshow("output_AttUnet", output_AttUnet)
            cv2.imshow("output_GANPlusAttUnet", output_GANPlusAttUnet)


            if Video_recording:
                self.videoWriter_mask.write(true_mask)
                self.videoWriter_GAN.write(output_Gan)
                self.videoWriter_Unet.write(output_Unet)
                self.videoWriter_AttUnet.write(output_AttUnet)
                self.videoWriter_GANPlusAttUnet.write(output_GANPlusAttUnet)
                self.videoWriter_img.write(img)


            ################### box analyse ############################
           
            cnts_GAN = cv2.findContours(output_Gan, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]  # im2,contours,hierarchy = cv.findContours(thresh, 1, 2)
            cnts_Unet = cv2.findContours(output_Unet, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
            cnts_AttUnet = cv2.findContours(output_AttUnet, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
            cnts_GANPlusAttUnet = cv2.findContours(output_GANPlusAttUnet, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
            cnts_Mask = cv2.findContours(true_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

            # minAreaRect
            self.vis_minAreaRect(img, output_Gan, cnts_GAN, color=(0, 0, 255)) # BGR red
            self.vis_minAreaRect(img, output_Unet, cnts_Unet, color=(0, 255, 0)) # BGR green
            self.vis_minAreaRect(img, output_AttUnet, cnts_AttUnet, color=(0, 125, 0)) # BGR light green
            self.vis_minAreaRect(img, output_GANPlusAttUnet, cnts_GANPlusAttUnet, color=(0, 125, 125)) # BGR light blue
            self.vis_minAreaRect(img, true_mask, cnts_Mask, color=(255, 0, 0)) # BGR blue

            # maxAreaRect
            self.vis_maxAreaRect(img, output_Gan, cnts_GAN, color=(0, 0, 255), model = "GAN")
            self.vis_maxAreaRect(img, output_Unet, cnts_Unet, color=(0, 255, 0), model="Unet")
            self.vis_maxAreaRect(img, output_AttUnet, cnts_AttUnet, color=(0, 255, 0), model="AttUnet")
            self.vis_maxAreaRect(img, output_GANPlusAttUnet, cnts_GANPlusAttUnet, color=(0, 255, 0), model="GANPlusAttUnet")
            self.vis_maxAreaRect(img, true_mask, cnts_Mask, color=(255, 0, 0), model="Target")

            
            ##############################################
            
            # ioU metric(based on mask and output)
            iou_gan = calculate_iou(true_mask, output_Gan)
            iou_unet = calculate_iou(true_mask, output_Unet)
            iou_attunet = calculate_iou(true_mask, output_AttUnet)
            iou_ganplusattunet = calculate_iou(true_mask, output_GANPlusAttUnet)
            # print("IOU score(GAN):", iou_gan)
            # print("IOU score(Unet):", iou_unet)
            iou_list_GAN.append(iou_gan)
            iou_list_Unet.append(iou_unet)
            iou_list_AttUnet.append(iou_attunet)
            iou_list_GANPlusAttUnet.append(iou_ganplusattunet)

            # dice metric list
            dice_list_GAN.append(dice(output_Gan, true_mask, k = 255))
            dice_list_Unet.append(dice(output_Unet, true_mask, k = 255))
            dice_list_AttUnet.append(dice(output_AttUnet, true_mask, k = 255))
            dice_list_GANPlusAttUnet.append(dice(output_GANPlusAttUnet, true_mask, k = 255))

            # final visualizaion
            cv2.imshow("current frame(red:GAN, green:Unet, blue:gt)", img)
            cv2.imshow("true_mask", true_mask)
            cv2.waitKey(10) 
            # cv2.waitKey(0)

        # print("dice_list:", dice_list)
        print("mean dice based GAN:", np.nanmean(dice_list_GAN))  # 需要使用nanmean忽略nan值,即忽略最前面的几帧
        print("mean dice based Unet:", np.nanmean(dice_list_Unet))
        print("mean dice based AttUnet:", np.nanmean(dice_list_AttUnet))
        print("mean dice based GANPlusAttUnet:", np.nanmean(dice_list_GANPlusAttUnet))

        print("mean iou based GAN:", np.nanmean(iou_list_GAN))
        print("mean iou based Unet:", np.nanmean(iou_list_Unet))
        print("mean iou based AttUnet:", np.nanmean(iou_list_AttUnet))
        print("mean iou based GANPlusAttUnet:", np.nanmean(iou_list_GANPlusAttUnet))


        if Video_recording:
            self.videoWriter_mask.release()
            self.videoWriter_GAN.release()
            self.videoWriter_Unet.release()
            self.videoWriter_AttUnet.release()
            self.videoWriter_GANPlusAttUnet.release()
            self.videoWriter_img.release()
        

    def vis_maxAreaRect(self, img, mask, cnts, color=(255, 0, 0), model = "GAN"):

        # 注意这里的img是对象,可以在这里直接画图，主循环中也会出现

        if len(cnts) != 0:
            cnt = sorted(cnts, key=cv2.contourArea, reverse=True)[0]  # only extract the max area contour,only one contour  
            x,y,w,h = cv2.boundingRect(cnt)
            img = cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
            
            # (x+w,y): right top point
            cv2.putText(img, model, (x+w-80,y-10), cv2.FONT_HERSHEY_COMPLEX, 1, color, 2) # (img, str,origin,font,size,color,thickness)


    def vis_minAreaRect(self, img, mask, cnts, color=(255, 0, 0)):

        if len(cnts) != 0:
            cnt = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
            rbox = cv2.minAreaRect(cnt)
            pts = cv2.boxPoints(rbox).astype(np.int32)
            cv2.drawContours(img, [pts], -1, color, 2, cv2.LINE_AA) 
        

if __name__ == "__main__":
    test_mode = "1" # 1/2 compounding 3/4 insertion
    eval = Evaluation(mode = test_mode)
    eval.start()