import cv2
import numpy as np
from scipy.spatial.distance import euclidean
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
from sklearn.decomposition import PCA
from scipy.ndimage import convolve

from sklearn.linear_model import LinearRegression

def outlier_filter(pred, mask, range = 5):

    '''According the mask, only extract the needle part from the pred image'''

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    exist_seg = False
    filtered_pred = None

    if len(contours) != 0:

        # Find the rotated rectangle for each contour
        rects = [cv2.minAreaRect(cnt) for cnt in contours]

        # For simplicity, let's take the first rectangle
        rect = rects[0]

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

        # Fill the area outside the rectangle with white
        cv2.fillPoly(mask_outside, [box], 255)

        # Bitwise-and with the pred image
        filtered_pred = cv2.bitwise_and(pred, mask_outside)

        # Display the result image
        # result_BGR = cv2.cvtColor(filtered_pred, cv2.COLOR_GRAY2BGR)
        # cv2.polylines(filtered_pred, [box], True, (0, 255, 0), 2)
        # cv2.imshow("Result", result_BGR)
        # cv2.imshow("Pred", pred)
        # cv2.imshow("Mask", mask)
        # cv2.waitKey(0)

        # safe check
        exist_seg = True

        
    return filtered_pred


def Calculate_Tip(image , mask):

    filter_image = outlier_filter(image, mask)

    cv2.imshow("image", image)
    cv2.imshow("filtered_pred", filter_image)
    cv2.waitKey(0)

    # 边缘检测
    edges = cv2.Canny(filter_image, 50, 150)

    # 提取边缘点的坐标
    y, x = np.nonzero(edges)
    points = np.column_stack((x, y))

    # 线性回归拟合
    model = LinearRegression()
    model.fit(points[:, 0].reshape(-1, 1), points[:, 1])

    # 计算倾斜角度
    slope = model.coef_[0]
    angle = np.arctan(slope) * 180 / np.pi

    # 计算靠右侧的端点
    x_sorted = np.sort(points[:, 0])
    y_pred = model.predict(x_sorted.reshape(-1, 1))
    rightmost_point = (x_sorted[-1], int(y_pred[-1]))

    return angle, rightmost_point


def Calculate_Continuity_LineProj(pred):

    '''Calculate the continuity of the needle by projecting the needle to the line'''
    # 找到针物体的轮廓

    image = outlier_filter(pred, mask)

    # 找到所有线段的轮廓
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 合并所有轮廓到一个点集中
    all_points = np.vstack([cnt.reshape(-1, 2) for cnt in contours])

    # 计算合并后的点集的旋转包围盒
    rect = cv2.minAreaRect(all_points)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # 计算旋转角度和中心点
    center, size, angle = rect
    if size[0] < size[1]:
        size = (size[1], size[0])
        angle += 90

    # 旋转图像
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    # 计算旋转后的包围盒位置
    x, y = np.int0(center)
    x, y = max(0, x - int(size[0] // 2)), max(0, y - int(size[1] // 2))
    cropped = rotated[y:y + int(size[1]), x:x + int(size[0])]
    
    cv2.imshow("cropped", cropped)
    cv2.waitKey(0)

    # 分析连续性 计算一个二维数组中每一列全黑（值为0）的列数
    black_count = np.sum(cropped == 0, axis=0) # 每一列的黑色像素数
    black_count = np.sum(black_count == cropped.shape[0])  # 全黑的列数
    # 计算连续性指标
    continuity = 1 - (black_count / size[0]) if size[0] > 0 else 1

    return continuity


# Load the image
# image_path = './data/test_dataset/2/masks/237.png' # debug for coundary case

# mask_path = './data/test_dataset/2/masks/265.png'
# pred_path = './data/sampleOut_GAN/65.png' # GAN sample
pred_path = './data/sampleOut_Unet/73.png' # UNet sample
mask_path = './data/test_dataset/2/masks/273.png'
pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
pred = cv2.normalize(pred, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
mask = cv2.normalize(mask, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)



print(Calculate_Tip(pred, mask))
print(Calculate_Continuity_LineProj(pred))
