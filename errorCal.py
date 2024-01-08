import cv2
import numpy as np
from scipy.spatial.distance import euclidean
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
from sklearn.decomposition import PCA
from scipy.ndimage import convolve
from sklearn.linear_model import LinearRegression
from scipy.spatial.distance import directed_hausdorff


image_H = 50
image_W = 51.3


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


class Calculate_Error():
        
    ''''Use the pred and mask to calculate the error(continuity and tip error))'''

    def __init__(self):


        pass

    def Calculate_TipError(self, pred = None, mask = None, method = 'canny'):

        '''To calculate the continuity, don't use filter_image'''
        _, exist_seg = self.outlier_filter(pred, mask) # 

        '''To calculate the tip error/angle error'''
        # needs some filter(outlier??)
        # filter_image_Tip = self.outlier_filter_outlier(pred)
        filter_image_Tip = pred
        

        if exist_seg == False:
            # only when the mask has a segment, the tip error/angle error start to calculate!!
            return float('NaN'), float('NaN')
        else:

            ang_pred, tip_pred = self.Calculate_TipOrientation(filter_image_Tip, method)
            ang_mask, tip_mask = self.Calculate_TipOrientation(mask, method)


            if np.isnan(ang_pred) or np.isnan(ang_mask):
                # if in this stage the angle is nan, it means the model failed to predict the angle
                return float('NaN'), float('NaN')
            
            else:

                # 计算角度误差
                angle_error = abs(ang_pred - ang_mask)

                # 计算位置误差
                tip_error = euclidean(tip_pred, tip_mask)

                return angle_error, tip_error

    def outlier_filter_outlier(self, pred, range = 10):
        '''According the pred structure, only extract the near needle part from the pred image'''
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

            # Create an all black image
            mask_outside = np.zeros_like(pred)

            # Fill the area inside the rectangle with white
            cv2.fillPoly(mask_outside, [box], 255)

            # Bitwise-and with the pred image
            filtered_pred = cv2.bitwise_and(pred, mask_outside)

            cv2.imshow("outlier filtered", filtered_pred)

        return filtered_pred

    def are_collinear(self, rect1, rect2, tolerance=3):
        # center, size, angle = rect
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

    
    def Calculate_TipOrientation(self, image, method):
        
        if method == 'canny':
            # 边缘检测
            edges = cv2.Canny(image, 250, 255)

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

        filtered_pred, exist_seg = self.outlier_filter(pred, mask) # Continuity calculation 

        if exist_seg == False:
            return float('NaN')
        else:
            if method == 'Box':
                return self.Calculate_Continuity_Box(filtered_pred)
            elif method == 'Projection':
                return self.Calculate_Continuity_projection(filtered_pred, mask)
            elif method == 'LineProj':
                return self.Calculate_Continuity_LineProj(filtered_pred, mask)
            elif method == 'Hausdorff':
                return self.Calculate_Continuity_Hausdorff(filtered_pred, mask)
            

    def Calculate_Continuity_LineProj(self, image, mask):

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

        # all_points = np.vstack([cnt.reshape(-1, 2) for cnt in contours]) # bug: if contours is empty, it will raise an error
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

    def extract_feature_points(self, image):
        # 这里以骨架化为例提取特征点
        skeleton = cv2.ximgproc.thinning(image)
        cv2.imshow("Skeleton", skeleton)
        cv2.waitKey(0)
        y, x = np.nonzero(skeleton)
        return np.column_stack((x, y))

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
    

    def outlier_filter_Tip(self, pred, mask, range = 5):

        '''According the previous frames, only extract the needle part from the predictions image'''



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

            
        return filtered_pred, exist_seg



    def outlier_filter(self, pred, mask, range = 5):

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

            # Display the result image
            # result_BGR = cv2.cvtColor(filtered_pred, cv2.COLOR_GRAY2BGR)
            # cv2.polylines(filtered_pred, [box], True, (0, 255, 0), 2)
            # cv2.imshow("Result", result_BGR)
            # cv2.imshow("Pred", pred)
            # cv2.imshow("Mask", mask)
            # cv2.waitKey(0)

            # safe check
            exist_seg = True

            
        return filtered_pred, exist_seg
    

    def visualize_principal_axis_and_projection(mask, prediction, axis, projected_points):

        """绘制主轴方向和投影点"""

        # 创建一个RGB图像用于可视化
        vis_image = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # 绘制主轴方向
        rows, cols = mask.shape
        center = np.mean(np.argwhere(mask > 0), axis=0).astype(np.int32)  # 使用mask的重心作为中心点
        axis_end = (int(center[1] + axis[0] * 100), int(center[0] + axis[1] * 100))  # 伸展轴以便可视化
        cv2.arrowedLine(vis_image, (center[1], center[0]), axis_end, (255, 0, 0), 2)

        # 绘制投影点
        for projection in projected_points:
            x, y = int(center[1] + projection * axis[0]), int(center[0] + projection * axis[1])
            cv2.circle(vis_image, (x, y), 1, (0, 255, 0), -1)  # 绿色点

        return vis_image
        
    def Calculate_Continuity_projection(self, pred, mask):

        # pred_BGR = cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR) # for visualization

        # 确定 mask 图像中线的主轴方向
        mask_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask_contour = max(mask_contours, key=cv2.contourArea)
        pca = PCA(n_components=2)
        pca.fit(mask_contour.reshape(-1, 2))

        # 得到主轴方向的单位向量
        principal_axis = pca.components_[0]

    # 将预测结果投影到主轴方向
        prediction_contours, _ = cv2.findContours(pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        projected_points = set()
        for contour in prediction_contours:
            for point in contour.reshape(-1, 2):
                projection = np.dot(point, principal_axis)
                projected_points.add(projection)

        # # 可视化主轴方向和投影点
        # vis_image = visualize_principal_axis_and_projection(mask, pred, principal_axis, projected_points)
        # cv2.imshow("Calculate_Continuity", vis_image)

        # 计算投影长度和覆盖长度
        if projected_points:

            projected_intervals = self.get_projected_intervals(projected_points, principal_axis, pca.mean_)

            # 计算覆盖的区间长度
            covered_length = self.calculate_covered_length(projected_intervals)
            

            # 计算总投影长度 (从 projected_intervals 列表中找出具有最大最小结束坐标的区间（Interval）)
            projection_length = max(projected_intervals, key=lambda interval: interval[1])[1] - \
                        min(projected_intervals, key=lambda interval: interval[0])[0]

            # 计算连续性指标
            continuity_index = covered_length / projection_length if projection_length > 0 else 1
        else:
            continuity_index = 1

        return continuity_index

    def get_projected_intervals(self, projected_points, axis, center, pixel_length=2):
        # 将每个点投影到主轴上的小区间
        intervals = []
        for point in projected_points:
            projection = np.dot(point - center, axis)
            intervals.append((projection - pixel_length / 2, projection + pixel_length / 2))

        return self.merge_intervals(intervals)

    def merge_intervals(self, intervals):
        # 合并重叠的区间
        sorted_intervals = sorted(intervals, key=lambda x: x[0])
        merged = []
        for interval in sorted_intervals:
            if not merged or merged[-1][1] < interval[0]:
                merged.append(interval)
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], interval[1]))

        return merged

    def calculate_covered_length(self, intervals):
        # 计算所有区间的总长度
        return sum(interval[1] - interval[0] for interval in intervals)
            



# Load the image
mask_path = './data/test_dataset/2/masks/273.png'
image_path = './data/test_dataset/2/images/273.png'

pred_path = './data/sampleOut_GAN/73.png' # GAN sample
# pred_path = './data/sampleOut_Unet/73.png' # UNet sample
pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

mask = cv2.normalize(mask, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
pred = cv2.normalize(pred, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)

calculate_object = Calculate_Error()

print(calculate_object.Calculate_Continuity(method = 'Projection', pred = pred, mask = mask))





