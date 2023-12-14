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


dir_checkpoint_GAN = 'C:/Research/projects/Learning/Needle_GANseg/analysis/RDB/fine_tune results/1/save_G_Exp/generator_18500.pth'