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
        # self.tf = Compose([Activations(sigmoid=True), Resize((657, 671)), AsDiscrete(threshold=0.5)])  # 先拉伸到原来的大小, 千万别忘了asdiscrete二值化
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


class Evaluation:
    
    def __init__(self, mode="1"):
        self.dataPath = './data/test_dataset/' + mode
        self.initialize_networks()
        self.load_data()
        self.initialize_video_writers(mode) if Video_recording else None

    def initialize_networks(self):
        self.net_GAN = NetworkInference_GANVer2("pork") 
        self.net_Unet = NetworkInference_Unet("pork", method="Unet")
        self.net_AttUnet = NetworkInference_Unet("pork", method="AttentionUnet")
        self.net_Deeplab = NetworkInference_DeepLabV3()
        self.net_DeeplabPlus = NetworkInference_DeepLabV3PLUS()

    def load_data(self):
        self.imgs = sorted(glob(self.dataPath + '/imgs/*.png'))
        self.masks = sorted(glob(self.dataPath + '/masks/*.png'))
        assert len(self.imgs) == len(self.masks), f'Image and mask counts do not match: {len(self.imgs)} vs {len(self.masks)}'

    def initialize_video_writers(self, mode):
        fps = 25
        img_size = (671, 657)
        fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
        self.videoWriters = {
            key: cv2.VideoWriter(f'./results/{key}_{mode}.avi', fourcc, fps, img_size, isColor=False)
            for key in ["mask", "GAN", "Unet", "AttUnet", "DeeplabV3", "DeeplabV3Plus"]
        }
        video_dir_img = './results/img_' + mode + '.avi'
        self.videoWriter_img = cv2.VideoWriter(video_dir_img, fourcc, fps, img_size, isColor=True) # 3 channels

    def start(self):
        print("All frames number:", len(self.imgs))
        metrics = {model: {'iou': [], 'dice': []} for model in ["GAN", "Unet", "AttUnet", "DeeplabV3", "DeeplabV3Plus"]}

        for i, (input, target) in enumerate(zip(self.imgs, self.masks)):
            img, true_mask = self.preprocess_images(input, target)
            outputs = self.generate_model_outputs(img)

            for key, output in outputs.items():
                self.visualize_and_record(key, img, true_mask, output)
                self.calculate_metrics(key, true_mask, output, metrics)

        self.print_metrics(metrics)
        self.release_video_writers() if Video_recording else None

    def preprocess_images(self, input_path, target_path):
        img = cv2.imread(input_path)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        true_mask = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE) * 255
        return img_gray, true_mask

    def generate_model_outputs(self, img):
        outputs = {
            "GAN": self.net_GAN.inference(img),
            "Unet": self.net_Unet.inference(img),
            "AttUnet": self.net_AttUnet.inference(img),
            "DeeplabV3": self.net_Deeplab.inference(img),
            "DeeplabV3Plus": self.net_DeeplabPlus.inference(img)
        }
        return {k: self.normalize_output(v) for k, v in outputs.items()}

    def normalize_output(self, output):
        return cv2.normalize(output, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)

    def visualize_and_record(self, model_name, img, true_mask, output):
        # Visualization and recording logic here
        pass

    def calculate_metrics(self, model_name, true_mask, output, metrics):
        iou = calculate_iou(true_mask, output)
        dice_score = dice(output, true_mask, k=255)
        metrics[model_name]['iou'].append(iou)
        metrics[model_name]['dice'].append(dice_score)

    def print_metrics(self, metrics):
        for model, values in metrics.items():
            mean_iou = np.nanmean(values['iou'])
            mean_dice = np.nanmean(values['dice'])
            print(f"mean iou for {model}: {mean_iou}")
            print(f"mean dice for {model}: {mean_dice}")

    def release_video_writers(self):
        for writer in self.videoWriters.values():
            writer.release()
        self.videoWriter_img.release()

    # Additional methods for visualization
    # ...

if __name__ == "__main__":
    test_mode = "3" # 1/2 compounding 3/4 insertion 
    eval = Evaluation(mode = test_mode)
    eval.start()
