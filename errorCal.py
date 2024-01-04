import cv2
import numpy as np
from scipy.spatial.distance import euclidean
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
from sklearn.decomposition import PCA
from scipy.ndimage import convolve



def Calculate_TipError(pred, mask):

    pass



class Calculate_Error():
        
    ''''Use the pred and mask to calculate the error(continuity and tip error))'''

    def __init__(self):

        # self.pred = pred
        # self.mask = mask
        # self.filtered_pred = self.outlier_filter(pred, mask)
        pass

    def Calculate_TipError(self):

        '''Calculate the tip position and angle error'''

        pass

    def Calculate_Continuity(self, method = 'Box', pred = None, mask = None):

        filtered_pred, exist_seg = self.outlier_filter(pred, mask)

        if exist_seg == False:
            return 0
        else:
            if method == 'Box':
                return self.Calculate_Continuity_Box(filtered_pred)
            elif method == 'Projection':
                return self.Calculate_Continuity_projection(filtered_pred, mask)

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





