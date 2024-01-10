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

from skimage import morphology
from skimage.measure import label, regionprops
from sklearn.linear_model import LinearRegression


def are_collinear(rect1, rect2, tolerance=3):
    # center, size, angle = rect
    # Calculate the difference in angles  
    angle_diff = abs(rect1[2] - rect2[2])
    
    # Calculate the slope and y-intercept of the line between the centers
    x_diff = rect2[0][0] - rect1[0][0]
    y_diff = rect2[0][1] - rect1[0][1]
    if abs(x_diff) < tolerance:
        slope = float('inf')
    else:
        slope = y_diff / x_diff

    y_intercept = rect1[0][1] - slope * rect1[0][0]
    
    # Calculate the angle of the line
    line_angle = np.degrees(np.arctan(slope)) if slope != float('inf') else 90
    
    # Check if the difference in angles is within the tolerance
    # 1. Check if the difference in angles is within the tolerance
    # 2. Check if the line is collinear with the rectangle
    return angle_diff < tolerance and abs(line_angle - rect1[2]) < tolerance

def outlier_filter_Tip(pred, range = 5, model="GAN"):

    '''According the pred structure, only extract the near needle part from the pred image'''

    # Find contours in the mask
    contours, _ = cv2.findContours(pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_pred = None

    if len(contours) != 0:

        # Find the rotated rectangle for each contour and calculate their areas
        rects = [(cv2.minAreaRect(cnt), cv2.contourArea(cnt)) for cnt in contours]

        # Sort the rectangles by area in descending order and take the first one
        rects.sort(key=lambda x: x[1], reverse=True)
        max_rect = rects[0][0] if rects else None

        # Initialize a list with the points of the maximum rectangle
        all_points = [cv2.boxPoints(max_rect)]

        # Iterate over the remaining rectangles
        for rect, _ in rects[1:]:
            # Check if the rectangle is collinear with the maximum rectangle
            if are_collinear(max_rect, rect):
                # If it is, add its points to the list
                all_points.append(cv2.boxPoints(rect))

        # Stack all the points into a single array
        all_points = np.vstack(all_points)

        # Calculate the minimum area rectangle that encloses all the points
        rect = cv2.minAreaRect(all_points)

        # Get the center, size, and angle from the rectangle
        center, size, angle = rect

        # Convert size to integer
        
        size = (int(size[0]) , int(size[1]) * range)

        # Get the four points of the rectangle
        rect = (center, size, angle)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # Create an all black image
        mask_outside = np.zeros_like(pred)

        # Fill the area inside the rectangle with white
        cv2.fillPoly(mask_outside, [box], 255)

        # Bitwise-and with the pred image
        filtered_pred = cv2.bitwise_and(pred, mask_outside)

        cv2.imshow("outlier filtered"+model, filtered_pred)

    return filtered_pred


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

class NetworkInference_GAN_FirstStage():
    '''
    Newest generator 
    '''
    def __init__(self, mode = "pork"):

        # dir_checkpoint_GAN = './test_model/FirstStage.pth'  # from first stage only using focal loss
        dir_checkpoint_GAN = './test_model/best_in6200_FirstStage.pth'
        
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
                self.tf = tf

            probs = self.tf(probs.cpu()) # 重新拉伸到原来的大小
            full_mask = probs.squeeze().cpu().numpy() # return in cpu  # 
            
            return full_mask


# Load the image
dataPath = "./data/test_dataset/optional/Alldatas/3/"
print("dataPath:", dataPath)
imgs = glob(dataPath + '/imgs' +"/*.png")
imgs.sort()

net_2 = NetworkInference_GANVer2()
net = NetworkInference_DeepLabV3PLUS()
net_3 = NetworkInference_Unet()
net_4 = NetworkInference_UnetPLUSPLUS()
net_5 = NetworkInference_GAN_FirstStage()
net_6 = NetworkInference_Unet(method = "AttentionUnet")

for i, input in enumerate(imgs):

    print("frame:", i)
    
    img = cv2.imread(input,0)
    img = cv2.normalize(img, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
    
    output = net.inference(img)
    output_2 = net_2.inference(img)
    output_3 = net_3.inference(img)
    output_4 = net_4.inference(img)
    output_5 = net_5.inference(img)
    output_6 = net_6.inference(img)

    output = cv2.normalize(output, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
    output_2 = cv2.normalize(output_2, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
    output_3 = cv2.normalize(output_3, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
    output_4 = cv2.normalize(output_4, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
    output_5 = cv2.normalize(output_5, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
    output_6 = cv2.normalize(output_6, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)


    # cv2.imshow("output_Deeplab", output)
    # cv2.imshow("output_GAN", output_2)
    # cv2.imshow("output_Unet", output_3)
    # cv2.imshow("output_Unet++", output_4)
    # cv2.imshow("output_FirstStage", output_5)
    # cv2.imshow("output_AttentionUnet", output_6)



    # outlier removal test

    outlier_filter_Tip(output, 10, "Deeplab")
    outlier_filter_Tip(output_2, 10, "GAN")
    outlier_filter_Tip(output_3, 10, "Unet")
    outlier_filter_Tip(output_4, 10, "Unet++")
    outlier_filter_Tip(output_5, 10, "FirstStage")
    outlier_filter_Tip(output_6, 10, "AttentionUnet")

    cv2.waitKey(1)


