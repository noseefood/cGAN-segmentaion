'''Used in windows environment'''

# from model.Generator import Generator # New structure  
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
import segmentation_models_pytorch as smp # DeepLabV3/DeepLabV3Plus

from errorCal import Calculate_Error

# whether to record the segmentation results
Video_recording = True

# Metrics
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

def calculate_TipError(gt_mask, pred_mask):
    """Calculate the tip error of the predicted mask(position and angle)"""
    pass

def calculate_ConError(gt_mask, pred_mask):
    pass

class NetworkInference_DeepLabV3():
    def __init__(self):
        # dir_checkpoint = "../Seg_baseline/results/DeeplabV3/best_metric_model_DeepLabV3.pth"
        dir_checkpoint = "./test_model/best_metric_model_DeepLabV3.pth"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = smp.DeepLabV3(    
            encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
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
                self.tf = tf

            probs = self.tf(probs.cpu()) # 重新拉伸到原来的大小
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

class Evaluation():
    
    def __init__(self, mode = "1"):

        self.dataPath = './data/test_dataset/' + mode
        self.net_GAN = NetworkInference_GANVer2("pork") 
        self.net_Unet = NetworkInference_Unet("pork", method = "Unet")  # "Unet" or "AttentionUnet" for comparison
        self.net_AttUnet = NetworkInference_Unet("pork", method = "AttentionUnet")  # "Unet" or "AttentionUnet" for comparison
        self.net_Deeplab = NetworkInference_DeepLabV3() # DeepLabV3
        self.net_DeeplabPlus = NetworkInference_DeepLabV3PLUS()

        
        # 
        self.Calculate_Error_object = Calculate_Error()

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
            video_dir_img = './results/img_' + mode + '.avi'
            video_dir_mask = './results/mask_' + mode + '.avi'
            video_dir_GAN = './results/GAN_' + mode + '.avi'
            video_dir_Unet = './results/Unet_' + mode + '.avi'
            video_dir_AttUnet = './results/AttUnet_' + mode + '.avi'
            video_dir_DeeplabV3 = './results/Deeplab_' + mode + '.avi'
            video_dir_DeeplabV3Plus = './results/DeeplabPlus_' + mode + '.avi'

            self.videoWriter_img = cv2.VideoWriter(video_dir_img, fourcc, fps, img_size, isColor=True) # 3 channels
            self.videoWriter_mask = cv2.VideoWriter(video_dir_mask, fourcc, fps, img_size, isColor=False)
            self.videoWriter_GAN = cv2.VideoWriter(video_dir_GAN, fourcc, fps, img_size, isColor=False)
            self.videoWriter_Unet = cv2.VideoWriter(video_dir_Unet, fourcc, fps, img_size, isColor=False)
            self.videoWriter_AttUnet = cv2.VideoWriter(video_dir_AttUnet, fourcc, fps, img_size, isColor=False)
            self.videoWriter_DeeplabV3 = cv2.VideoWriter(video_dir_DeeplabV3, fourcc, fps, img_size, isColor=False)
            self.videoWriter_DeeplabV3Plus = cv2.VideoWriter(video_dir_DeeplabV3Plus, fourcc, fps, img_size, isColor=False)

    def start(self):

        assert len(self.imgs) == len(self.masks), \
            f'len(self.imgs) should be equal to len(self.masks), but got {len(self.imgs)} vs {len(self.masks)}'

        print("All frames number:", len(self.imgs))

        dice_list_GAN = []
        dice_list_Unet = []
        dice_list_AttUnet = []
        dice_list_DeeplabV3 = []
        dice_list_DeeplabV3Plus = []

        iou_list_GAN = []
        iou_list_Unet = []
        iou_list_AttUnet = []
        iou_list_DeeplabV3 = []
        iou_list_DeeplabV3Plus = []

        for i,(input, target) in enumerate(zip(self.imgs,self.masks)):
            
            img = cv2.imread(input)
            true_mask = cv2.imread(target) 
            true_mask = cv2.cvtColor(true_mask, cv2.COLOR_BGR2GRAY) * 255 # 0-1 -> 0-255

            output_Gan = self.net_GAN.inference(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)) # 0-1
            output_Unet = self.net_Unet.inference(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)) # 0-1
            output_AttUnet = self.net_AttUnet.inference(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)) # 0-1
            output_DeeplabV3 = self.net_Deeplab.inference(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)) # 0-1
            output_DeeplabV3Plus = self.net_DeeplabPlus.inference(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)) # 0-1

            output_Gan = cv2.normalize(output_Gan, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U) # 0-1 -> 0-255
            output_Unet = cv2.normalize(output_Unet, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U) # 0-1 -> 0-255
            output_AttUnet = cv2.normalize(output_AttUnet, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U) # 0-1 -> 0-255
            output_DeeplabV3 = cv2.normalize(output_DeeplabV3, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U) # 0-1 -> 0-255
            output_DeeplabV3Plus = cv2.normalize(output_DeeplabV3Plus, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U) # 0-1 -> 0-255

            # print("Image data type:", output_Gan.dtype)
            # print("Image data type:", output_Unet.dtype)

            cv2.imshow("output_GAN", output_Gan)    
            cv2.imshow("output_Unet", output_Unet)
            cv2.imshow("output_AttUnet", output_AttUnet)
            cv2.imshow("output_DeeplabV3", output_DeeplabV3)
            cv2.imshow("output_DeeplabV3Plus", output_DeeplabV3Plus)


            print(self.Calculate_Error_object.Calculate_Continuity(method = 'Projection', pred = output_Gan, mask = true_mask))


            # # save image for analysis
            # cv2.imwrite("./data/sampleOut_Unet/" + str(i) + ".png", output_Unet)
            


            if Video_recording:
                self.videoWriter_mask.write(true_mask)
                self.videoWriter_GAN.write(output_Gan)
                self.videoWriter_Unet.write(output_Unet)
                self.videoWriter_AttUnet.write(output_AttUnet)
                self.videoWriter_DeeplabV3.write(output_DeeplabV3)
                self.videoWriter_DeeplabV3Plus.write(output_DeeplabV3Plus)
                self.videoWriter_img.write(img)


            ################### box analyse ############################
           
            cnts_GAN = cv2.findContours(output_Gan, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]  # im2,contours,hierarchy = cv.findContours(thresh, 1, 2)
            cnts_Unet = cv2.findContours(output_Unet, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
            cnts_AttUnet = cv2.findContours(output_AttUnet, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
            cnts_DeeplabV3 = cv2.findContours(output_DeeplabV3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

            cnts_Mask = cv2.findContours(true_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

            # minAreaRect
            self.vis_minAreaRect(img, output_Gan, cnts_GAN, color=(0, 0, 255)) # BGR red
            self.vis_minAreaRect(img, output_Unet, cnts_Unet, color=(0, 255, 0)) # BGR green
            self.vis_minAreaRect(img, output_AttUnet, cnts_AttUnet, color=(0, 125, 0)) # BGR light green
            self.vis_minAreaRect(img, output_DeeplabV3, cnts_DeeplabV3, color=(0, 125, 125))

            self.vis_minAreaRect(img, true_mask, cnts_Mask, color=(255, 0, 0)) # BGR blue

            # maxAreaRect
            self.vis_maxAreaRect(img, output_Gan, cnts_GAN, color=(0, 0, 255), model = "GAN")
            self.vis_maxAreaRect(img, output_Unet, cnts_Unet, color=(0, 255, 0), model="Unet")
            self.vis_maxAreaRect(img, output_AttUnet, cnts_AttUnet, color=(0, 255, 0), model="AttUnet")
            self.vis_maxAreaRect(img, output_DeeplabV3, cnts_DeeplabV3, color=(0, 255, 0), model="DeeplabV3")

            self.vis_maxAreaRect(img, true_mask, cnts_Mask, color=(255, 0, 0), model="Target")

            
            ##############################################
            
            # ioU metric(based on mask and output)
            iou_gan = calculate_iou(true_mask, output_Gan)
            iou_unet = calculate_iou(true_mask, output_Unet)
            iou_attunet = calculate_iou(true_mask, output_AttUnet)
            iou_deeplabv3 = calculate_iou(true_mask, output_DeeplabV3)
            iou_deeplabv3plus = calculate_iou(true_mask, output_DeeplabV3Plus)


            # print("IOU score(GAN):", iou_gan)
            # print("IOU score(Unet):", iou_unet)
            iou_list_GAN.append(iou_gan)
            iou_list_Unet.append(iou_unet)
            iou_list_AttUnet.append(iou_attunet)
            iou_list_DeeplabV3.append(iou_deeplabv3)
            iou_list_DeeplabV3Plus.append(iou_deeplabv3plus)


            # dice metric list
            dice_list_GAN.append(dice(output_Gan, true_mask, k = 255))
            dice_list_Unet.append(dice(output_Unet, true_mask, k = 255))
            dice_list_AttUnet.append(dice(output_AttUnet, true_mask, k = 255))
            dice_list_DeeplabV3.append(dice(output_DeeplabV3, true_mask, k = 255))
            dice_list_DeeplabV3Plus.append(dice(output_DeeplabV3Plus, true_mask, k = 255))


            # final visualizaion
            cv2.imshow("current frame(red:GAN, green:Unet, blue:gt)", img)
            cv2.imshow("true_mask", true_mask)
            cv2.waitKey(10) 
            # cv2.waitKey(0)

        # print("dice_list:", dice_list)
        print("mean dice based GAN:", np.nanmean(dice_list_GAN))  # 需要使用nanmean忽略nan值,即忽略最前面的几帧
        print("mean dice based Unet:", np.nanmean(dice_list_Unet))
        print("mean dice based AttUnet:", np.nanmean(dice_list_AttUnet))
        print("mean dice based DeeplabV3:", np.nanmean(dice_list_DeeplabV3))
        print("mean dice based DeeplabV3Plus:", np.nanmean(dice_list_DeeplabV3Plus))


        print("mean iou based GAN:", np.nanmean(iou_list_GAN))
        print("mean iou based Unet:", np.nanmean(iou_list_Unet))
        print("mean iou based AttUnet:", np.nanmean(iou_list_AttUnet))
        print("mean iou based DeeplabV3:", np.nanmean(iou_list_DeeplabV3))
        print("mean iou based DeeplabV3Plus:", np.nanmean(iou_list_DeeplabV3Plus))



        if Video_recording:
            self.videoWriter_mask.release()
            self.videoWriter_GAN.release()
            self.videoWriter_Unet.release()
            self.videoWriter_AttUnet.release()
            self.videoWriter_DeeplabV3.release()
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
    test_mode = "2" # 1/2 compounding 3/4 insertion 
    eval = Evaluation(mode = test_mode)
    eval.start()