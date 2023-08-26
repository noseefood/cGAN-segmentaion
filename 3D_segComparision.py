from model.Generator import Generator
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
import SimpleITK as sitk


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

    def inference(self, img, resize_tf = None):
        with torch.no_grad():
            img = self.train_imtrans(img) # compose会自动返回tensor torch.Size([1, 512, 512])

            img = img.to(self.device) # torch.Size([1, 512, 512])   HWC to CHW：img_trans = img_nd.transpose((2, 0, 1))
            img = img.unsqueeze(0) # torch.Size([1, 1, 512, 512]) unsqueeze扩增维度
            
            # # 因为这里没有使用dataloader读取，所以不需要转置，输出可以直接与原图对比
            # img = img.transpose(-1,-2) # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 注意这里与inference.py不同，这里不需要转置，以为读取图片的方式不一样！

            output = self.model(img)
            result = self.post_trans(output) # torch.Size([1, 1, 512, 512])

            probs = result.squeeze(0) # squeeze压缩维度 torch.Size([1, 512, 512])
            
            if resize_tf is not None:
                # self.tf = resize_tf
                self.tf = Compose([Resize(resize_tf),])

            probs = self.tf(probs.cpu()) # 重新拉伸到原来的大小

            full_mask = probs.squeeze().cpu().numpy() # return in cpu  # 

            return full_mask


class NetworkInference_GAN():
    
    def __init__(self, mode = "pork"):

        dir_checkpoint_GAN = '/home/xuesong/CAMP/segment/cGAN-segmentaion/data/models/05.08/runs/generator_26500.pth'
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.generator = Generator().to(self.device)  # input channel = 1
        # self.generator.load_state_dict(torch.load(dir_checkpoint + "UAGAN_generator.pth"))
        self.generator.load_state_dict(torch.load(dir_checkpoint_GAN))
        self.generator.eval() # eval mode

        self.train_imtrans = Compose( # 预处理
            [   
                AddChannel(),  # 增加维度
                Resize((480, 480)), # 必须要加入这个，否则会报错，这里相当于直接拉伸，跟training保持一致
                ScaleIntensity(), # 归一化 0-255 -> 0-1
            ]
        )
        # self.post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])  # 这里不需要sigmoid，因为网络中已经有了sigmoid
        self.tf = Compose([Resize((657, 671)), AsDiscrete(threshold=0.5)])  # 先拉伸到原来的大小, 别忘了asdiscrete二值化

    def inference(self, img, resize_tf = None):
        
        with torch.no_grad():
            
            # TODO:检测输入图片的通道数，如果是3通道，需要转换为1通道

            img = self.train_imtrans(img) # compose会自动返回tensor torch.Size([1, 512, 512])

            img = img.to(self.device) # torch.Size([1, 512, 512])   HWC to CHW：img_trans = img_nd.transpose((2, 0, 1))
            img = img.unsqueeze(0) # torch.Size([1, 1, 512, 512]) unsqueeze扩增维度
            # # 因为这里没有使用dataloader读取，所以不需要转置，输出可以直接与原图对比
            # img = img.transpose(-1,-2) # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 注意这里与inference.py不同，这里不需要转置，以为读取图片的方式不一样！

            output = self.generator(img)

            probs = output.squeeze(0) # squeeze压缩维度 torch.Size([1, 512, 512])

            if resize_tf is not None:
                self.tf = Compose([Resize((resize_tf)), AsDiscrete(threshold=0.5)])

            probs = self.tf(probs.cpu()) # 重新拉伸到原来的大小
            full_mask = probs.squeeze().cpu().numpy() # return in cpu  # 
            # cv2.imshow("full_mask", full_mask)
            
            return full_mask
        

class VoxelSeg():
    def __init__(self, itkimage_unSeg):
        # self.voxelImg_unSeg = self.itk2voxel(itkimage_unSeg)
        self.itkimage_unSeg = itkimage_unSeg
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.voxelImg_unSeg = self.itk2voxel(itkimage_unSeg)
        self.voxelImg_SegGAN= self.voxelImg_unSeg.copy()
        self.voxelImg_SegUnet = self.voxelImg_unSeg.copy()

        self.net_GAN = NetworkInference_GAN("pork")
        self.net_Unet = NetworkInference_Unet("pork", method = "AttentionUnet")  # "Unet" or "AttentionUnet" for comparison


    def itk2voxel(self, itkImg):
        # itk image to numpy array
        temp = sitk.GetArrayFromImage(itkImg)   # (154, 418, 449) z,y,x 转换为numpy (z,y,x): z:切片数量,y:切片宽,x:切片高
        return temp


    def process_and_replace_slices(self):
        # 获取图像的尺寸
        image = self.itkimage_unSeg
        output_file_path_Unet = "/home/xuesong/CAMP/segment/selfMonaiSegment/data/3D_seg/Unet_segmented_Volume.mhd"
        output_file_path_GAN = "/home/xuesong/CAMP/segment/selfMonaiSegment/data/3D_seg/GAN_segmented_Volume.mhd"
        size = image.GetSize()  # (449, 418, 154) x,y,z 注意直接读取itkimage(xyz)和转换为numpy(zyx)的区别 

        # self.model.eval()  no need, implemented in inference function
        # with torch.no_grad():

        for z in range(size[2]):
            # 提取单个切片
            slice_filter = sitk.ExtractImageFilter()
            slice_filter.SetSize([size[0], size[1], 0])
            slice_filter.SetIndex([0, 0, z])
            slice_image = slice_filter.Execute(image) # itk image

            # 在这里对切片进行修改，您可以添加您的图像处理代码
            img = sitk.GetArrayFromImage(slice_image)
            cv2.imshow("original slice img", img)
            size_1, size_2 = img.shape
            resize_tf = (size_1, size_2)

            output_Gan = self.net_GAN.inference(img, resize_tf)
            output_Unet = self.net_Unet.inference(img, resize_tf)
            output_Gan = cv2.normalize(output_Gan, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U) # 0-1 -> 0-255
            output_Unet = cv2.normalize(output_Unet, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)

            self.voxelImg_SegGAN[z,:,:] = output_Gan 
            self.voxelImg_SegUnet[z,:,:] = output_Unet


        # 保存结果
        replaced_image_GAN = sitk.GetImageFromArray(self.voxelImg_SegGAN)
        replaced_image_GAN.CopyInformation(self.itkimage_unSeg)
        sitk.WriteImage(replaced_image_GAN, output_file_path_GAN)

        replaced_image_Unet = sitk.GetImageFromArray(self.voxelImg_SegUnet)  
        replaced_image_Unet.CopyInformation(self.itkimage_unSeg) # 注意这里暂时还是itk image
        sitk.WriteImage(replaced_image_Unet, output_file_path_Unet)  # z保存为mhd文件



itkimage_unSeg = sitk.ReadImage("/home/xuesong/CAMP/segment/selfMonaiSegment/data/3D_seg/unsegmented_Volume.mhd")
extractor = VoxelSeg(itkimage_unSeg)
extractor.process_and_replace_slices()