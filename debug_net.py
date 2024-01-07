from model.Generator_ReLU import Generator
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
import segmentation_models_pytorch as smp


def calculate_metrics(mask, prediction):

    # 二值化处理
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    _, prediction = cv2.threshold(prediction, 127, 255, cv2.THRESH_BINARY)

    # 计算混淆矩阵
    TP = np.sum((mask == 255) & (prediction == 255))
    TN = np.sum((mask == 0) & (prediction == 0))
    FP = np.sum((mask == 0) & (prediction == 255))
    FN = np.sum((mask == 255) & (prediction == 0))

    # 计算召回率和精确度
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0

    # 计算 F2 分数
    F2 = (1 + 2**2) * precision * recall / (2**2 * precision + recall) if (precision + recall) != 0 else 0

    return recall, precision, F2

class NetworkInference_UnetPLUSPLUS():
    def __init__(self):
        dir_checkpoint = "./test_model/best_metric_model_Unet++.pth"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = smp.UnetPlusPlus(    
            encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for en coder initialization
            in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,).to(self.device)                  # a number of channels of output mask
        self.model.load_state_dict(torch.load(dir_checkpoint))
        self.model.eval()

        self.imgs_preprocess = Compose( # Pre-processing
            [   
                AddChannel(),  # 增加channel
                Resize((512, 512)), # 跟training保持一致
                ScaleIntensity(), # 0-255 -> 0-1
            ])
        self.imgs_postprocess = (Compose([Activations(sigmoid=True),Resize((657, 671)), AsDiscrete(threshold=0.5)]) )

    def inference(self, img, tf = None):
        with torch.no_grad():

            img = self.imgs_preprocess(img) # compose会自动返回tensor torch.Size([1, 512, 512])

            img = img.to(self.device) # torch.Size([1, 512, 512])   HWC to CHW：img_trans = img_nd.transpose((2, 0, 1))
            img = img.unsqueeze(0) # torch.Size([1, 1, 512, 512]) unsqueeze扩增维度

            output = self.model(img)
            probs = output.squeeze(0) # squeeze压缩维度 torch.Size([1, 512, 512])

            if tf is not None:
                self.imgs_postprocess = tf

            probs = self.imgs_postprocess(probs.cpu()) # 重新拉伸到原来的大小
            full_mask = probs.squeeze().cpu().numpy() # return in cpu  # 
            
            return full_mask




class NetworkInference_GANVer2():
    '''
    Newest generator 
    '''
    def __init__(self, mode = "pork"):

        # dir_checkpoint_GAN = './test_model/best_model_in7000.pth' # from Colab Jan03_13-16
        dir_checkpoint_GAN = './test_model/model_in6000_Jan02_15-55.pth' # from Colab Jan02_15-55
        # dir_checkpoint_GAN = './test_model/FirstStage.pth'  # from first stage only using focal loss
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.generator = Generator().to(self.device)  # input channel = 1
        self.generator.load_state_dict(torch.load(dir_checkpoint_GAN))
        self.generator.eval() # eval mode

        self.train_imtrans = Compose( # Pre-processing
            [   
                AddChannel(),  # 增加channel
                Resize((512, 512)), # 跟training保持一致
                ScaleIntensity(), # 0-255 -> 0-1
            ]
        )
        # self.tf = Compose([Activations(sigmoid=True), Resize((657, 671)), AsDiscrete(threshold=0.5)])  # 先拉伸到原来的大小, 别忘了asdiscrete二值化
        self.tf = Compose([Activations(sigmoid=True),Resize((657, 671)), AsDiscrete(threshold=0.5)]) 

    def inference(self, img, tf = None):
        
        with torch.no_grad():
            
            # TODO:检测输入图片的通道数，如果是3通道，需要转换为1通道

            img = self.train_imtrans(img) # compose会自动返回tensor torch.Size([1, 512, 512])

            img = img.to(self.device) # torch.Size([1, 512, 512])   HWC to CHW：img_trans = img_nd.transpose((2, 0, 1))
            img = img.unsqueeze(0) # torch.Size([1, 1, 512, 512]) unsqueeze扩增维度

            output = self.generator(img)
            probs = output.squeeze(0) # squeeze压缩维度 torch.Size([1, 512, 512])

            if tf is not None:
                self.tf = Compose([Activations(sigmoid=True),Resize(tf), AsDiscrete(threshold=0.5)]) 

            probs = self.tf(probs.cpu()) # 重新拉伸到原来的大小
            full_mask = probs.squeeze().cpu().numpy() # return in cpu  # 
            
            return full_mask
        
class NetworkInference_DeepLabV3PLUS():
    def __init__(self):
        # dir_checkpoint = "../Seg_baseline/results/DeeplabV3PLUS/best_metric_model_DeepLabV3PLUS.pth" # Source
        dir_checkpoint = "./test_model//best_metric_model_DeepLabV3PLUS.pth"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = smp.DeepLabV3Plus(    
            encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for en coder initialization
            in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,).to(self.device)                  # a number of channels of output mask
        self.model.load_state_dict(torch.load(dir_checkpoint))
        self.model.eval()

        self.imgs_preprocess = Compose( # Pre-processing
            [   
                AddChannel(),  # 增加channel
                Resize((512, 512)), # 跟training保持一致
                ScaleIntensity(), # 0-255 -> 0-1
            ])
        self.imgs_postprocess = (Compose([Activations(sigmoid=True),Resize((657, 671)), AsDiscrete(threshold=0.5)]) )

    def inference(self, img, tf = None):
        with torch.no_grad():

            img = self.imgs_preprocess(img) # compose会自动返回tensor torch.Size([1, 512, 512])

            img = img.to(self.device) # torch.Size([1, 512, 512])   HWC to CHW：img_trans = img_nd.transpose((2, 0, 1))
            img = img.unsqueeze(0) # torch.Size([1, 1, 512, 512]) unsqueeze扩增维度

            output = self.model(img)
            probs = output.squeeze(0) # squeeze压缩维度 torch.Size([1, 512, 512])

            if tf is not None:
                self.imgs_postprocess = tf

            probs = self.imgs_postprocess(probs.cpu()) # 重新拉伸到原来的大小
            full_mask = probs.squeeze().cpu().numpy() # return in cpu  # 
            
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
            self.model.load_state_dict(torch.load("./test_model/best_metric_model_AttentionUnet.pth"))

        elif method == "Unet":
            self.model = UNet(
                spatial_dims=2,
                in_channels=1,
                out_channels=1,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
                num_res_units=2,
            ).to(self.device)
            self.model.load_state_dict(torch.load("./test_model/best_metric_model_Unet.pth"))

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
            
            return full_mask


# Load the image
# dataPath = "/home/xuesong/CAMP/dataset/datasetTest_080823/addtional_060124/5/"
dataPath = "./data/test_dataset/2/"
print("dataPath:", dataPath)
imgs = glob(dataPath + '/imgs' +"/*.png")
masks = glob(dataPath + '/masks' +"/*.png") 
imgs.sort()
masks.sort()

net_2 = NetworkInference_GANVer2()
net = NetworkInference_DeepLabV3PLUS()
net_3 = NetworkInference_Unet()
net_4 = NetworkInference_UnetPLUSPLUS()

for i, (input,mask) in enumerate(zip(imgs, masks)):
    
    img = cv2.imread(input,0)
    img = cv2.normalize(img, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
    mask = cv2.imread(mask,0)
    mask = cv2.normalize(mask, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
    
    output = net.inference(img)
    output_2 = net_2.inference(img)
    output_3 = net_3.inference(img)
    output_4 = net_4.inference(img)

    output = cv2.normalize(output, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
    output_2 = cv2.normalize(output_2, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
    output_3 = cv2.normalize(output_3, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
    output_4 = cv2.normalize(output_4, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)

    recall, precision, F2 = calculate_metrics(mask, output)
    print("recall:", recall, "precision:", precision, "F2:", F2)

    cv2.imshow("output", output)
    cv2.imshow("output_GAN", output_2)
    cv2.imshow("output_Unet", output_3)
    cv2.imshow("output_Unet++", output_4)
    cv2.imshow("mask", mask)
    cv2.waitKey(0)



# img = cv2.imread('/home/xuesong/Tmech_imgs/MethodsComparision/test/test.png', cv2.IMREAD_GRAYSCALE)
# img = cv2.normalize(img, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)

# output = net.inference(img)

# cv2.imshow("output", output)
# cv2.waitKey(0)
