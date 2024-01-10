'''
Calculating metrics for all models in 2D US images, mainly for Insertion mode
Dice metric; iou metric; F2 metric; Precision; Recall; Continuity; TipError; AngleError
'''
from glob import glob
import cv2
import time
import numpy as np
from Metric_cal import Calculate_Error, calculate_metrics, calculate_iou, dice
import pandas as pd

from models import NetworkInference_GANVer2, NetworkInference_Unet, NetworkInference_DeepLabV3, NetworkInference_DeepLabV3PLUS, NetworkInference_UnetPLUSPLUS, NetworkInference_GAN_FirstStage
from SegModul3D import VoxelSeg, ExtractModul


import SimpleITK as sitk
import pyvista

colinear_filter = True
visualizaion = True
print_results = True

class Evaluation():
    
    def __init__(self, mode = "Compounding", dataset = 1):
        
        self.dataset = dataset

        # initialize models
        self.net_GAN = NetworkInference_GANVer2("pork") 
        self.net_Unet = NetworkInference_Unet("pork", method = "Unet")  
        self.net_AttUnet = NetworkInference_Unet("pork", method = "AttentionUnet")  
        self.net_DeeplabPlus = NetworkInference_DeepLabV3PLUS()
        self.net_UnetPlusPlus = NetworkInference_UnetPLUSPLUS()
        self.net_GAN_FirstStage = NetworkInference_GAN_FirstStage("pork") # for first stage only using focal loss

        # initialize Calculate_Error object
        self.Calculate_Error_object = Calculate_Error(colinear_processing = colinear_filter)
        self.colinear_used = "Colinear" if colinear_filter else "NoColinear"

        # initialize data
        self.mode = mode
        self.dataPath = './data/test_dataset/' + str(dataset) + '/unsegmented_Volume.mhd'
        self.itkimage_unSeg = sitk.ReadImage(self.dataPath)
        print("dataPath:", self.dataPath)


    def start(self):

        models = {
            'GAN': self.net_GAN,
            'Unet': self.net_Unet,
            'AttUnet': self.net_AttUnet,
            'DeeplabV3Plus': self.net_DeeplabPlus,
            'UnetPlusPlus': self.net_UnetPlusPlus,
            'GAN_FirstStage': self.net_GAN_FirstStage
        }

        # ######################## iterate all models ##########################

        for model_name, model in models.items():
            print(f'Processing model {model_name} ...')
            self.segmentation3D(self.itkimage_unSeg, model, model_name, method = "RANSAC")


    def segmentation3D(self, itkimage_unSeg, model, model_name, method = "RANSAC"):
        '''
        Directly excecuting 3D segmentation and further extracting needle pose for debug-simulation or automatic mode
        This code is directly copied from the robotic repo's client_gui.py
        '''
        time_start = time.time()

        # directly segment the volume
        seg_3D = VoxelSeg(itkimage_unSeg, model, method="RANSAC") # RANSAC, Mittelpoint and Projection 
        itkimage_Seg = seg_3D.segment3D() # 保存分割结果mhd同时返回itk_image
        
        # extrcat the needle pose from the segmented volume
        extract_modul = ExtractModul(itkimage_Seg, "itkimage", thresthold = 1, voxel_size = 2)

        if method == "RANSAC":
            # nb_neighbors = 100, std_ratio = 0.1 default value
            extract_modul.preprocess() # Statistic outlier removal
        elif method == "Mittelpoint":
            # nb_neighbors = 100, std_ratio = 0.1
            extract_modul.preprocess(nb_neighbors=100, std_ratio=0.1) # Statistic outlier removal

        point_1, point_2 = extract_modul.extract()

        # extract the needle physical parameter from the volume 
        point1_base = itkimage_unSeg.TransformContinuousIndexToPhysicalPoint(point_1)  # 返回的是tuple  TransformPhysicalPointToIndex则可以将空间坐标转换为体素坐标
        point2_base = itkimage_unSeg.TransformContinuousIndexToPhysicalPoint(point_2)
        time_end = time.time()
        # save the results for comparision




        print('time cost for 3D segmentation and extraction', time_end - time_start, 's')






        # save
        sitk.WriteImage(itkimage_Seg, './results/segmented_Volume' + model_name + str(self.dataset) +'.mhd')


        
        
        # extract_modul = ExtractModul(itkimage_Seg, "itkimage", 1)
        # extract_modul.preprocess()
        # point_1, point_2 = extract_modul.extract()
        # # print("points in voxel coordinate: ", point_1, point_2)

        # # extract the needle physical parameter from the volume 
        # point1_world = itkimage_unSeg.TransformContinuousIndexToPhysicalPoint(point_1)  # 返回的是tuple  TransformPhysicalPointToIndex则可以将空间坐标转换为体素坐标
        # point2_world = itkimage_unSeg.TransformContinuousIndexToPhysicalPoint(point_2)

        # print("point_1 in world coordinate: ", point1_world)  
        # print("point_2 in world coordinate: ", point2_world)

        # time_end = time.time()
        # print('time cost for 3D segmentation and extraction', time_end - time_start, 's')


if __name__ == "__main__":
    test_mode = "Compounding" # "Compounding"
    dataset = 1 # 1,5
    eval = Evaluation(mode = test_mode, dataset = dataset)
    eval.start()