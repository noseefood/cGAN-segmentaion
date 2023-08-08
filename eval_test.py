from model.Generator import Generator
import os
import sys
from glob import glob
import cv2

import torch
from PIL import Image

from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    LoadImage,
    ScaleIntensity,
    Resize,
    AddChannel,
)
from monai.metrics import DiceMetric

from pathlib import Path
base = Path(os.environ['raw_data_base']) if 'raw_data_base' in os.environ.keys() else Path('./data')
assert base is not None, "Please assign the raw_data_base(which store the training data) in system path "
dir_test = base / 'test/test_1'
# dir_checkpoint = '/home/xuesong/CAMP/segment/UAGAN/save_model/save_G/'
dir_checkpoint = '/home/xuesong/CAMP/segment/cGAN-segmentaion/data/models/05.08/runs/generator_26500.pth'


import time

class NetworkInference():
    def __init__(self, mode = "pork"):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.generator = Generator().to(self.device)  # input channel = 1
        # self.generator.load_state_dict(torch.load(dir_checkpoint + "UAGAN_generator.pth"))
        self.generator.load_state_dict(torch.load(dir_checkpoint))
        self.generator.eval() # eval mode


        self.train_imtrans = Compose( # 输入模型的图片的预处理
            [   
                AddChannel(),  # 增加通道维度
                Resize((480, 480)), # 必须要加入这个，否则会报错，这里相当于直接拉伸，跟training保持一致
                ScaleIntensity(), # 其实就是归一化
            ]
        )
        # self.post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])  # 这里不需要sigmoid，因为网络中已经有了sigmoid
        self.tf = Compose([Resize((657, 671)),])  # 重新拉伸到原来的大小

    def inference(self, img):
        
        with torch.no_grad():

            img = self.train_imtrans(img) # compose会自动返回tensor torch.Size([1, 512, 512])

            img = img.to(self.device) # torch.Size([1, 512, 512])   HWC to CHW：img_trans = img_nd.transpose((2, 0, 1))
            img = img.unsqueeze(0) # torch.Size([1, 1, 512, 512]) unsqueeze扩增维度
            
            # # 因为这里没有使用dataloader读取，所以不需要转置，输出可以直接与原图对比
            # img = img.transpose(-1,-2) # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 注意这里与inference.py不同，这里不需要转置，以为读取图片的方式不一样！

            output = self.generator(img)
            # result = self.post_trans(output) # torch.Size([1, 1, 512, 512])

            probs = output.squeeze(0) # squeeze压缩维度 torch.Size([1, 512, 512])
            probs = self.tf(probs.cpu()) # 重新拉伸到原来的大小
            full_mask = probs.squeeze().cpu().numpy() # return in cpu  # 
            
            # cv2.imshow("full_mask", full_mask)
            
            return full_mask



class Evaluation():
    def __init__(self, mode = "water"):

        if mode == "water": # water
            self.VideoCap = cv2.VideoCapture('/home/xuesong/CAMP/dataset/video_sim/water_tank.avi')
            self.net = NetworkInference("water")
        elif mode == "pork-missaligen": # insertion-needle
            self.VideoCap = cv2.VideoCapture('/home/xuesong/CAMP/dataset/video_sim/pork_insertion_newProk17.avi')
            self.net = NetworkInference("pork")
        elif mode == "pork-3Dsegmentaion": # compounding-needle
            self.VideoCap = cv2.VideoCapture('/home/xuesong/CAMP/dataset/video_sim/pork_compounding.avi')
            self.net = NetworkInference("pork")

        self.num_frames =  self.VideoCap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.frame_counter = 0 
        self.ControlSpeedVar = 50
        self.HiSpeed = 100

    def start(self):

        print("All frames number:", self.num_frames)

        while(True):

            ret, frame = self.VideoCap.read()

            self.frame_counter += 1
            if self.frame_counter == int(self.VideoCap.get(cv2.CAP_PROP_FRAME_COUNT)):
                self.frame_counter = 0
                self.VideoCap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            ###############################
            # 2D filter
            time_start = time.time()
            mask = self.net.inference(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))  # from 3-channels into 1-channel Gray
            time_end = time.time()
            print('totally cost', time_end-time_start, ' seconds')

            cv2.imshow('frame', frame)
            cv2.imshow('mask', mask)
            ###############################

            k = cv2.waitKey(30) & 0xff
            if k == 27: # ESC key to exit
                break
            cv2.waitKey(self.HiSpeed-self.ControlSpeedVar+1)
        cv2.destroyAllWindows()
        self.VideoCap.release()

if __name__ == "__main__":
    US_mode = "pork-missaligen" # "water" "pork-missaligen" "pork-3Dsegmentaion"
    eval = Evaluation(mode = US_mode)
    eval.start()