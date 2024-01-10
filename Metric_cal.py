import cv2
import numpy as np
# import collections
from scipy.spatial.distance import euclidean
from skimage.morphology import skeletonize
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from scipy.spatial.distance import directed_hausdorff

image_H = 50
image_W = 51.3
pixel_H = 657 # ratio 0.076 mm/pixel
pixel_W = 671 # ratio 0.078 mm/pixel, average 0.077 mm/pixel
ratioPixel2mm = 0.077
# buffer_number = 50

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

def calculate_metrics(mask, prediction):

    # 二值化处理
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    _, prediction = cv2.threshold(prediction, 127, 255, cv2.THRESH_BINARY)

    # 计算混淆矩阵
    TP = np.sum((mask == 255) & (prediction == 255))
    TN = np.sum((mask == 0) & (prediction == 0))
    FP = np.sum((mask == 0) & (prediction == 255))
    FN = np.sum((mask == 255) & (prediction == 0))

    # 计算召回率和精确度
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0

    # 计算 F2 分数
    F2 = (1 + 2**2) * precision * recall / (2**2 * precision + recall) if (precision + recall) != 0 else 0

    return recall, precision, F2

def pixel2mm(pixel):
    '''
    Convert the pixel to mm
    2D US: Depth: 50mm, Width: 51.3mm
    '''
    pixel_H = pixel[0]
    pixel_W = pixel[1]
    mm_H = pixel_H / image_H * 50
    mm_W = pixel_W / image_W * 51.3

    return mm_H, mm_H

class Post_processing():
    '''
    Post-processing the pred image, every model has its own post-processing object.
    It has the following functions:
        1. Basic outlier_filter: According the colinear principle[OK]
        2. Enhandced outlier_filter: According the previous frames[Not OK]
            principle: in needle insertion, the other part of the image is static, only the needle part is moving,
                        so we can use the previous frames to filter the needle part.
            two prequesites: 1. this box is very small
                             2. this box stays in the same position for a very long time
    '''
    def __init__(self, pred = None, mdoel_name = None):
        self.model_Name = mdoel_name
        # self.buffer_ = collections.deque(maxlen=buffer_number) # append

    def basic_outlier_filter(self, pred, range = 5):
        '''
        According the pred structure, only extract the near needle part from the pred image
        pred: the pred image
        range: the range of merged rectangle width
        '''
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
                if self.are_collinear(max_rect, rect):
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

            # filter
            mask_outside = np.zeros_like(pred)
            cv2.fillPoly(mask_outside, [box], 255)
            filtered_pred = cv2.bitwise_and(pred, mask_outside)

            # cv2.imshow("outlier filtered" + self.model_Name, filtered_pred)

        return filtered_pred


    def are_collinear(self, rect1, rect2, tolerance=3):
        '''
        Check if two rectangles are collinear
        rect1/2: (center, size, angle)
        tolerance: the tolerance of the angle difference
        '''
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
    

    def enhandced_outlier_filter(self, pred):

        pass

    def compounding_outlier_filter(self, pred):
        '''
        
        '''



class Calculate_Error():
    '''
    Using the pred to calculate the error(angle error and tip error)
    Using the pred and mask to calculate the continuity
    '''

    def __init__(self, colinear_processing = True):
         
        self.post_processing = Post_processing()
        self.colinear_processing = colinear_processing

    def Calculate_TipError(self, pred = None, mask = None, method = 'canny'):

        # Only use mask as start signal
        _, exist_seg = self.outlier_filter(pred, mask) # 


        # Post-processing to filter the outlier
        if self.colinear_processing:
            filter_img = self.post_processing.basic_outlier_filter(pred)
        else:
            filter_img = pred
        

        if exist_seg == False:
            # only when the mask has a segment, the tip error/angle error start to calculate!!
            return float('NaN'), float('NaN'), float('NaN'), pred
        
        else:
            ang_pred, tip_pred = self.Calculate_TipOrientation(filter_img, method)
            ang_mask, tip_mask = self.Calculate_TipOrientation(mask, method)

            dist_img = self.calculate_dist_center(filter_img)
            dist_mask = self.calculate_dist_center(mask)


            if np.isnan(ang_pred) or np.isnan(ang_mask):
                # if in this stage the angle is nan, it means the model failed to predict the angle
                return float('NaN'), float('NaN'), float('NaN'), filter_img

            else:
                # 计算角度误差(都是与水平的夹角,degree)
                angle_error = abs(ang_pred - ang_mask)

                # 计算位置误差(pixel)
                tip_error = euclidean(tip_pred, tip_mask)

                # 计算距离误差(for angle error,pixel)
                frame_center_error = euclidean(dist_img, dist_mask)
                
                # 转换为mm
                tip_error = tip_error * ratioPixel2mm
                frame_center_error = frame_center_error * ratioPixel2mm

                # print("angle_error: ", angle_error)
                # print("tip_error: ", tip_error)
                # print("frame_center_error: ", frame_center_error)
                # print("filter_img: ", filter_img)

                return angle_error, tip_error, frame_center_error, filter_img

    
    def Calculate_TipOrientation(self, image, method):
        
        if method == 'canny':
            # edge detection
            edges = cv2.Canny(image, 250, 255)
            
        if edges is not None:

            if edges.size != 0:
                
                # 提取边缘点的坐标
                y, x = np.nonzero(edges)
                points = np.column_stack((x, y))
                # print(points)

                if points.size == 0 :
                    return float('NaN'), (float('NaN'), float('NaN'))
                    print("Empty points !")
                
                else:

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
            else:
                return float('NaN'), (float('NaN'), float('NaN'))
        else:
            return float('NaN'), (float('NaN'), float('NaN'))
            
        if method == 'threshold':
            # 提取特征点
            y, x = np.nonzero(image)

            # 线性回归拟合
            model = LinearRegression()
            model.fit(x.reshape(-1, 1), y)

            # 计算倾斜角度
            slope = model.coef_[0]
            angle = np.arctan(slope) * 180 / np.pi

            # 找到靠右侧的端点
            x_sorted = np.sort(x)
            rightmost_x = x_sorted[-1]
            rightmost_y = model.predict([[rightmost_x]])[0]

            return angle, (rightmost_x, rightmost_y)
        
        if method == 'skeleton':

            skeleton = skeletonize(image // 255).astype(np.uint8)

            # 提取骨架点
            y, x = np.nonzero(skeleton)

            # 线性回归拟合
            model = LinearRegression()
            model.fit(x.reshape(-1, 1), y)

            # 计算倾斜角度
            slope = model.coef_[0]
            angle = np.arctan(slope) * 180 / np.pi

            # 找到靠右侧的端点
            x_sorted = np.sort(x)
            rightmost_x = x_sorted[-1]
            rightmost_y = model.predict([[rightmost_x]])[0]

            return angle, (rightmost_x, rightmost_y)


    def Calculate_Continuity(self, method = 'LineProj', pred = None, mask = None):
        
        # Posteriori filter using mask, only for analysis of continuity's computation!!!!!!!!1
        filtered_pred, exist_seg = self.outlier_filter(pred, mask) # Continuity calculation 


        if exist_seg == False:
            return float('NaN')
        else:
            if method == 'Box':
                return self.Calculate_Continuity_Box(filtered_pred)
            elif method == 'LineProj':
                return self.Calculate_Continuity_LineProj(filtered_pred)
            elif method == 'Hausdorff':
                return self.Calculate_Continuity_Hausdorff(filtered_pred, mask)
            

    def Calculate_Continuity_LineProj(self, image):

        '''Calculate the continuity of the needle by projecting the needle to the line'''

        thickness_factor = 1

        # 找到所有线段的轮廓
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 合并所有轮廓到一个点集中
        # 只有当filter之后的prediction存在contours时,才有必要计算连续性,否则直接为0!!!
        if contours:
            all_points = np.vstack([cnt.reshape(-1, 2) for cnt in contours])
            # 计算合并后的点集的旋转包围盒
            rect = cv2.minAreaRect(all_points)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            # 计算旋转角度和中心点
            center, size, angle = rect
            if size[0] < size[1]:
                size = (size[1], size[0] * thickness_factor)
                angle += 90
            else:
                size = (size[0], size[1] * thickness_factor)

            # 旋转图像
            M = cv2.getRotationMatrix2D(center, angle, 1.0) # Scale = 1.0
            rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

            # 计算旋转后的包围盒位置
            x, y = np.int0(center)
            x, y = max(0, x - int(size[0] // 2)), max(0, y - int(size[1] // 2))
            cropped = rotated[y:y + int(size[1]), x:x + int(size[0])]
            
            # cv2.imshow("cropped", cropped)
            # cv2.waitKey(0)

            # 分析连续性 计算一个二维数组中每一列全黑（值为0）的列数
            black_count = np.sum(cropped == 0, axis=0) # 每一列的黑色像素数
            # black_count = np.sum(black_count == cropped.shape[0])  # 全黑的列数
            black_count = np.sum(black_count >= cropped.shape[0] * 0.7) #
            # 计算连续性指标
            continuity = 1 - (black_count / size[0]) if size[0] > 0 else 1

        else:
            continuity = 0

        return continuity

    def Calculate_Continuity_Box(self, pred):

        # pred_BGR = cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR) # for visualization

        # 计算和可视化 prediction 中每个线段的矩形框
        prediction_contours, _ = cv2.findContours(pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        total_box_area = 0
        x_min, y_min = np.inf, np.inf
        x_max, y_max = -np.inf, -np.inf
        for cnt in prediction_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            total_box_area += w * h
            # cv2.rectangle(pred_BGR, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green boxes
            x_min, y_min = min(x_min, x), min(y_min, y)
            x_max, y_max = max(x_max, x + w), max(y_max, y + h)

        # 基于所有线段的最大尺寸计算和可视化 mask 的矩形框
        mask_max_box_area = (x_max - x_min) * (y_max - y_min)
        # cv2.rectangle(pred_BGR, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)  # Blue box

        # 计算连续性指标
        continuity_index = total_box_area / mask_max_box_area if mask_max_box_area > 0 else 0

        # visualize the result
        # cv2.imshow("Calculate_Continuity", pred_BGR)
        # cv2.waitKey(0)

        return continuity_index

    def Calculate_Continuity_Hausdorff(self, pred, mask):
        # 提取特征点
        points_mask = self.extract_feature_points(mask)
        points_predicted = self.extract_feature_points(pred)

        # 计算豪斯多夫距离
        hausdorff_dist = max(directed_hausdorff(points_mask, points_predicted)[0],
                            directed_hausdorff(points_predicted, points_mask)[0])

        # 转换为连续性得分
        continuity_score = hausdorff_dist

        return continuity_score

    def extract_feature_points(self, image):
        # 这里以骨架化为例提取特征点
        skeleton = cv2.ximgproc.thinning(image)
        cv2.imshow("Skeleton", skeleton)
        cv2.waitKey(0)
        y, x = np.nonzero(skeleton)
        return np.column_stack((x, y))

    def outlier_filter(self, pred, mask, range = 5):
        '''
        According the mask, only extract the needle part from the pred image
        Only mask used in continuity calculation
        '''
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
            # int(size[0])这个方向有必要扩大吗??
            
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

            # safe check
            exist_seg = True
            
        return filtered_pred, exist_seg
    
    def calculate_dist_center(self, pred):
        '''
        To calcluate the distance between the center of every frame and the line
        It also can be used to calculate the relative angle between the pred and the mask
        '''

        edges = cv2.Canny(pred, 250, 255)

        if edges is not None:

            if edges.size != 0:

                # 提取边缘点的坐标
                y, x = np.nonzero(edges)
                points = np.column_stack((x, y))
                slope,intercept = 0, 0

                if points.size != 0 :
                    # 线性回归拟合
                    model = LinearRegression()
                    model.fit(points[:, 0].reshape(-1, 1), points[:, 1])

                    # 计算倾斜角度
                    slope,intercept = model.coef_[0], model.intercept_
                    angle = np.arctan(slope) * 180 / np.pi


                # Calculate the distance between the center of every frame and the line
                # Assuming 'image' is your 2D image
                    
                # check the image shape
                if len(pred.shape) == 3:
                    height, width, _ = pred.shape   
                else:
                    height, width = pred.shape

                # Calculate the center pixel coordinates
                h, k  = height // 2, width // 2

                # Calculate distance from center to the line
                A = -slope
                B = 1
                C = -intercept

                distance = abs(A*h + B*k + C) / np.sqrt(A**2 + B**2)

                return distance

    def Compounding_3D_error(self, pred, true_value):
        pass
