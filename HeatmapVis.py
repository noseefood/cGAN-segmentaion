from pytorch_grad_cam import GradCAMPlusPlus, GradCAM
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from model.Generator_ReLU import Generator 
from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

# from monai.transforms import (
#     Activations,
#     AsDiscrete,
#     Compose,
#     ScaleIntensity,
#     Resize,
#     AddChannel,
# )

from torchvision import transforms
from monai.networks.nets import UNet, AttentionUnet

def main():
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("running on the GPU")
    else:
        device = torch.device("cpu")
        print("running on the CPU")

    # model = AttentionUnet(
    #             spatial_dims=2,
    #             in_channels=1,
    #             out_channels=1,
    #             channels=(16, 32, 64, 128, 256),
    #             strides=(2, 2, 2, 2),  
    #             kernel_size=3,
    # ).to(device)
    # dir_checkpoint_Att = './test_model/best_metric_model_AttentionUnet.pth'
    # model.load_state_dict(torch.load(dir_checkpoint_Att))

    model = Generator().to(device)
    dir_checkpoint_GAN = './test_model/best_model_in10900.pth'
    model.load_state_dict(torch.load(dir_checkpoint_GAN))

    model.eval()
    target_layers = [model.out_layer[-1]]

    # img_path = "../Needle_GANseg/data/Data_Pork/imgs/US_02_0062.png"
    img_path = "../dataset/test_dataset/3/imgs/3075.png"
    input_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    # input_img = cv2.normalize(input_img, None, 1, 0, cv2.NORM_MINMAX, cv2.CV_8U)

    data_transform_pre = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((512, 512)),
    ])
    data_transform_post = transforms.Compose([
        # transforms.Resize((657, 671)),
    ])

    img_tensor = data_transform_pre(input_img).to(device) # [C, H, W]
    input_tensor = torch.unsqueeze(img_tensor, dim=0) # [N, C, H, W] [1, 1, 512, 512]
    
    output = model(input_tensor)
    output = torch.sigmoid(output) # 0-1
    # output = data_transform_post(output) # [C, H, W] [1, 512, 512]

    output = output.squeeze(0).cpu().detach().numpy()  # [C, H, W] [1, 512, 512]
    # output = 


    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
    targets = [SemanticSegmentationTarget(0, output)]
    grayscale_cam = cam(input_tensor=input_tensor, targets = targets) #!!!!


    grayscale_cam = grayscale_cam[0, :]
    print(grayscale_cam.shape) # (512, 512)
    grayscale_cam_uint8 = cv2.normalize(grayscale_cam, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    grayscale_cam_mask = np.float32(grayscale_cam)
    cv2.imshow('heatmap.png', grayscale_cam)
    cv2.waitKey(0)
    plt.imshow(grayscale_cam, cmap='jet')
    plt.show()

    vis_input = cv2.resize(input_img, (512, 512), interpolation= cv2.INTER_LINEAR)
    vis_input = cv2.normalize(vis_input, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
    vis_input = np.expand_dims(vis_input, axis=-1)
    vis_input = np.repeat(vis_input, 3, axis=-1)

    visualization = show_cam_on_image(vis_input, 
                                      grayscale_cam,
                                      use_rgb=True)
    
    plt.imshow(visualization, cmap='jet')
    plt.show()

if __name__ == "__main__":
    main()