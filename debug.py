import cv2
import numpy as np
from scipy.spatial.distance import euclidean
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
from sklearn.decomposition import PCA
from scipy.ndimage import convolve

def Calculate_Tip(image):

    # Find contours
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Assume the largest contour is the needle and find its endpoints
    needle_contour = max(contours, key=cv2.contourArea) # find the largest contour
    rows, cols = image.shape

    # # Create a black image
    # image_largest_contour = np.zeros_like(image)

    # # Draw only the largest contour
    # cv2.drawContours(image_largest_contour, [needle_contour], -1, (255), thickness=cv2.FILLED)

    # # Skeletonize the image with only the largest contour
    # skeleton = skeletonize(image_largest_contour // 255)

    # 骨架是对象的简化表示，仅保留对象的中心线
    skeleton = skeletonize(image // 255)  # // 255 to convert from binary to boolean 

    # Old method
    # # Find the endpoints of the skeleton
    # endpoints = []
    # for y in range(rows):
    #     for x in range(cols):
    #         if skeleton[y, x]: # If we are at a pixel belonging to the skeleton
    #             # 如果该像素是骨架的一部分，那么代码计算该像素点的8邻域中有多少像素同样是骨架的一部分。这是通过对以 (y, x) 为中心的3x3区域内的像素值求和来实现的。
    #             neighbours = np.sum(skeleton[y-1:y+2, x-1:x+2])
    #             # Endpoint will have exactly one neighbour 在骨架中，端点像素除了它自己外只会有一个相邻的像素
    #             if neighbours == 2:
    #                 endpoints.append((x, y))

    # New method
    kernel = np.array([[1, 1, 1], 
                    [1, 10, 1],  # 中心像素的权重设置得比邻居大
                    [1, 1, 1]])

    # 对骨架图像进行卷积，计算邻居总数
    neighbour_count = convolve(skeleton.astype(np.uint8), kernel, mode='constant', cval=0)

    # 找到端点：在骨架上且邻居总数等于11的点（10来自自身，1来自一个邻居）
    endpoints = np.column_stack(np.where(neighbour_count == 11))

    cv2.imshow("skeleton", skeleton.astype(np.uint8) * 255)




    # If we have exactly two endpoints, we can use them to find the angle
    # Otherwise, we might need additional logic to determine which are the endpoints
    if len(endpoints) == 2:
        # Calculate the angle of the needle
        (x1, y1), (x2, y2) = endpoints
        angle_rad = np.arctan2(y2 - y1, x2 - x1)
        angle_deg = np.degrees(angle_rad)

        # Determine which point is the tip of the needle
        # Assuming that the tip is the uppermost point (smallest y value)
        tip = min(endpoints, key=lambda point: point[1])
    else:
        angle_deg = None
        tip = None

    cv2.imshow("mask", image)
    

    # Create an RGB version of the gray image
    color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Overlay the skeleton on the image by changing the color of skeleton pixels
    color_image[skeleton == 1] = [0, 0, 255]  # Red color

    # Display the overlay image
    cv2.imshow('Skeleton overlay', color_image)
    cv2.waitKey(0)

    return angle_deg, tip, endpoints



# Load the image
# image_path = './data/test_dataset/2/masks/268.png'
# image_path = './data/sampleOut_GAN/65.png' # GAN sample
image_path = './data/sampleOut_Unet/73.png' # UNet sample
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

image = cv2.normalize(image, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)


angle_deg, tip, endpoints = Calculate_Tip(image)
print(angle_deg, tip, endpoints)