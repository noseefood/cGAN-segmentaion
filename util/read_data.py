import os
import cv2 as cv
import torch
import numpy as np
import random

import albumentations as A  

random.seed(777)

class SegmentationDataset(object):
    def __init__(self, image_dir, mask_dir):
        self.images = []
        self.masks = []
        files = os.listdir(image_dir)
        sfiles = os.listdir(mask_dir)
        for i in range(len(sfiles)):
            img_file = os.path.join(image_dir, files[i])
            mask_file = os.path.join(mask_dir, sfiles[i])
            self.images.append(img_file)
            self.masks.append(mask_file)

        # augmentation
        
        # self.transform = A.Compose([
        #             A.HorizontalFlip(p=0.3),
        #             A.VerticalFlip(p=0.3),
        #             A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        #             A.GaussNoise(p=0.2),
        #             # A.CLAHE(p=0.2), # Contrast Limited Adaptive Histogram Equalization
        #             A.GaussianBlur(p=0.2),
        #             # A.PiecewiseAffine(p=0.2, scale=(0.03, 0.04), nb_rows=(4, 4), nb_cols=(4, 4)),  # new feature colab not support...
        #             # A.ShiftScaleRotate(p=0.3, shift_limit=0.0625, scale_limit=0.1, rotate_limit=20, border_mode=cv.BORDER_CONSTANT, value=0, mask_value=0),
        #         ])
        self.transform = A.Compose([    # water
                    A.HorizontalFlip(p=0.3),
                    A.VerticalFlip(p=0.3),
                    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
                    # A.GaussNoise(p=0.2),
                    # A.CLAHE(p=0.2), # Contrast Limited Adaptive Histogram Equalization
                    # A.GaussianBlur(p=0.2),
                ])

    def __len__(self):
        return len(self.images)

    def num_of_samples(self):
        return len(self.images)
    
    # def transform(self, image, mask):
    #     pass

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            image_path = self.images[idx]
            mask_path = self.masks[idx]
        else:
            image_path = self.images[idx]
            mask_path = self.masks[idx]
            
        img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)  # BGR order
        mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
        

        ########### 原始代码这里没有加resize 这个模型是480*480输入的, for AttentionUNet是512*512
        img = cv.resize(img, (512, 512))
        mask = cv.resize(mask, (512, 512))
        ###############################

        transformed = self.transform(image=img, mask=mask)  # 必须一起输入才能保证对img/mask统一的变换
        img = transformed["image"]
        mask = transformed["mask"]        

        # img = cv.cvtColor(img,cv.COLOR_GRAY2RGB)

        # 输入图像 preprocessing
        img = np.float32(img) / 255.0
        img = np.expand_dims(img, 0) # add channel dim from 
        mask = np.expand_dims(mask, 0) # add channel dim

        sample = {'image': torch.from_numpy(img), 'mask': torch.from_numpy(mask), }


        return sample
