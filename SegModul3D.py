import numpy as np
import SimpleITK as sitk
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    Resize,
)
import cv2
from Metric_cal import Post_processing
import open3d as o3d
import pyransac3d as pyrsc

using_colinear_filter = False
debug_flag = True
mirror_img = True   


class VoxelSeg():
    def __init__(self, itkimage_unSeg, net, method="RANSAC"):
        """
        method: how to obtaion needle-line parameters, para: RANSAC, Mittelpoint
        RANSAC: purely based on RANSAC for segmented result
        Mittelpoint: after extracting the mittelpoint of each segmented frame and then excuting RANSAC
        """
        self.post_processing = Post_processing() # Basic img filter
        self.itkimage_unSeg = itkimage_unSeg
        self.voxelImg_unSeg = self.itk2voxel(itkimage_unSeg) # np array
        self.voxelImg_Seg = self.voxelImg_unSeg.copy() # innitialization

        self.method = method
        self.net = net  # 直接从外部传入网络实例,都已经eval了并且no_grad了

    def itk2voxel(self, itkImg):
        '''
        itk image to numpy array
        '''
        temp = sitk.GetArrayFromImage(itkImg)   # (154, 418, 449) z,y,x 转换为numpy (z,y,x): z:切片数量,y:切片宽,x:切片高
        
        return temp
    
    def segment3D(self):
        """
        segmentation 3D volume interface
        """
        segmentVol = None

        segmentVol = self.process_and_replace_slices()

        return segmentVol


    def process_and_replace_slices(self):
        """
        process every single slice and replace the original slice with the segmented result
        RANSAC, Mittelpoint implementation here
        """

        # acquire the itk image and size
        image = self.itkimage_unSeg
        size = image.GetSize()  # (449, 418, 154) x,y,z 注意直接读取itkimage(xyz)和转换为numpy(zyx)的区别 

        for z in range(size[2]):
            # extract every single slice
            slice_filter = sitk.ExtractImageFilter()
            slice_filter.SetSize([size[0], size[1], 0])
            slice_filter.SetIndex([0, 0, z])
            slice_image = slice_filter.Execute(image) # itk image


            # slice
            img = sitk.GetArrayFromImage(slice_image)
            size_1, size_2 = img.shape
            resize_tf = (size_1, size_2)
            cv2.imshow("original", img)

            # # mirror
            # if mirror_img:
            #     img = cv2.flip(img, 1)
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U) # 0-1 -> 0-255
            # cv2.imshow("original", img)
            

            # inference
            imgs_postprocess = (Compose([Activations(sigmoid=True),Resize(resize_tf), AsDiscrete(threshold=0.5)]) )
            output = self.net.inference(img, imgs_postprocess)
            output = cv2.normalize(output, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U) # 0-1 -> 0-255

            # cv2.imshow("segmented", output)
            cv2.waitKey(1)

            # post processing
            if using_colinear_filter:
                output = self.post_processing.basic_outlier_filter(output)

            if self.method == "RANSAC":
                self.voxelImg_Seg[z,:,:] = output # slice repalce
            elif self.method == "Mittelpoint":
                output_copy = output.copy()
                point_img = self.mittelpoint(output_copy)
                self.voxelImg_Seg[z,:,:] = point_img # slice repalce
        
        print("3D segmentation finished!")

        # save the segmented result(for archive debug)
        replaced_image = sitk.GetImageFromArray(self.voxelImg_Seg)
        replaced_image.CopyInformation(self.itkimage_unSeg)
        # sitk.WriteImage(replaced_image, output_file_path) # 会保存为mhd文件,只用于存档保存方便后续分析

        return replaced_image # segmented itk image
    
    def mittelpoint(self, input):
        """input: single segmented image(0-255)"""
        """Mittelpoint method kernel"""
        h, w = input.shape  # one channel
        blank_image = np.zeros((h, w), np.uint8)

        cnts = cv2.findContours(input, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

        if len(cnts) != 0:
            # '''只使用最大的轮廓'''
            # cnt = sorted(cnts, key=cv2.contourArea, reverse=True)[0]  # only extract the max area contour,only one contour 
            # x,y,w,h = cv2.boundingRect(cnt)
            # cv2.circle(blank_image, (x+w//2,y+h//2), radius=1, color=(255,255,255), thickness=2) # need to remove segmentaion outlier and not write in the same image with RANSAC 
            
            """!!!!!!!!!!!注意,这里实际上应该设置为遍历所有的cnt,防止出现局部的误分割导致的needle提取错误!!!!!!!!!!!""" 
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
            for cnt in cnts:
                x,y,w,h = cv2.boundingRect(cnt)
                cv2.circle(blank_image, (x+w//2,y+h//2), radius=1, color=(255,255,255), thickness=2)

        return blank_image
    




class ExtractModul():
    '''
    Additional function for paper:
    1. extract the needle tip(simplest way: the lowest point in the volume/pcd) and calculate the needle pose(two angles)
    '''

    def __init__(self, input, input_type, thresthold, spacing=0.12, voxel_size=2, parent=None):
        self.transform = None
        self.thresthold = thresthold
        self.pcd_data = None
        self.voxel_data = None
        self.voxel_range = None
        self.voxel_size = voxel_size # downsample voxel size

        if input_type == "itkimage":
            # itk image下不需要spacing,只需要关注体素坐标就足够了,点云坐标即体素索引坐标
            self.voxel_range = input.GetSize() # z,y,x 这里不要在意顺序,因为转换到物理坐标系之后会自动转换
            self.voxel_data = self.itk2voxel(input) # z,y,x
            self.pcd_data = self.voxel2pcd(self.voxel_data)
            
        elif input_type == "ImFusion":
            '''
            [imfusion.SharedImageSet(size: 1, [imfusion.SharedImage(UBYTE width: 511 height: 539 slices: 464 spacing: 0.12 mm)])]  
            '''
            temp = self.imf2voxel(input)
            self.voxel_data = temp
            self.pcd_data = self.voxel2pcd(temp)
            self.transform = input.matrixToWorld(-1)

        elif input_type == "Voxel":
            # H*W*D numpy array
            self.voxel_data = input
            self.pcd_data = self.voxel2pcd(input)

        elif input_type == "Pointcloud":
            # N*3 numpy array
            self.pcd_data = input

        # self.pcd_data = self.pcd_data * spacing ################# 很确定肯定是xyz的个共同系数
            
        assert self.pcd_data.ndim == 2 and self.pcd_data.shape[1] == 3, \
            "The input data should be N*3 numpy array! current shape: {}".format(self.pcd_data.shape)
        
        # convert to open3d pointcloud
        self.open3d_pcd = o3d.geometry.PointCloud()
        self.open3d_pcd.points = o3d.utility.Vector3dVector(self.pcd_data)
        self.open3d_pcd = o3d.geometry.PointCloud.voxel_down_sample(self.open3d_pcd, self.voxel_size)  # voxel_size: 1mm
        

        
    def extract(self, analysis=False):

        '''open3d_pcd在这里是已经被filter过了的点云(statistical_outlier_removal或者DBSCAN)'''
        
        # line ransac(pyRANSAC-3D)
        points = np.asarray(self.open3d_pcd.points)

        # RANSAC 3D
        line = pyrsc.Line() # thresh：内点的距离阈值
        A, B, inliers = line.fit(points, thresh=5, maxIteration=1000) # thresh需要调整,A: 3D slope of the line (angle),B: Axis interception of the line

        # two representative points on the needle-line
        slope = A
        interception = (B[0], B[1], B[2])

        if analysis == True:
            # for analysis
            '''
            slope : 3D slope of the line (angle) [1,3]
            '''
            Tip_point = self.extract_needle_tip(inliers, slope) # list
            print("Tip_point: ", Tip_point)
            
            degrees = self.slope_to_degrees(slope)
            print("degrees: ", degrees)




            return Tip_point, degrees

        else:
            # for experiment
            point_1, point_2 = self.get_BoundPoints(interception, slope, self.voxel_range)  # Along the needle-line to get the two points on the bound surfaces of Volume(Sweep); self.voxel_range) in z,y,x
            # print("slope, interception of Needle-line:" , slope, interception)
            # print("Extracted two points(point_1, point_2): ", point_1, point_2)

            # visualization in open3d
            plane = self.open3d_pcd.select_by_index(inliers).paint_uniform_color([1, 0, 0])
            plane.paint_uniform_color([0,0,1])

            R = pyrsc.get_rotationMatrix_from_vectors([0, 0, 1], A)
            mesh_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=1, height=1000)
            mesh_cylinder.compute_vertex_normals()
            mesh_cylinder.paint_uniform_color([1, 0, 0])
            mesh_cylinder = mesh_cylinder.rotate(R, center=[0, 0, 0])
            mesh_cylinder = mesh_cylinder.translate((B[0], B[1], B[2]))
            mesh_cylinder.paint_uniform_color([0,1,0])

            point_1_open3d = self.create_sphere_at_xyz(point_1)  
            point_2_open3d = self.create_sphere_at_xyz(point_2)
            origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=20, origin=[0, 0, 0]) # frame 坐标点[x,y,z]对应R，G，B颜色

            if debug_flag:
                o3d.visualization.draw_geometries([self.open3d_pcd, plane, mesh_cylinder, point_1_open3d, point_2_open3d, origin])

            return point_1, point_2
    
    def extract_needle_tip(self, inliers, slope):
        '''
        `inliers`: Inlier's index from the original point cloud. `np.array (1, M)`
        '''

        pcd_inInlier = self.open3d_pcd.select_by_index(inliers)
        points = np.asarray(pcd_inInlier.points)

        tip_index = np.argmax(points[:, 1])

        point_tip = points[tip_index] # list [x,y,z]
        # print("tip_index: ", point_tip) 

        # visualization 
        point_tip_o3d = self.create_sphere_at_xyz(point_tip) 

        if debug_flag:
            origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=20, origin=[0, 0, 0]) # frame 坐标点[x,y,z]对应R，G，B颜色
            o3d.visualization.draw_geometries([self.open3d_pcd, point_tip_o3d, origin, pcd_inInlier])

        return point_tip
    
        
    def slope_to_degrees(self, slope):
        # Normalize the slope vector
        slope = slope / np.linalg.norm(slope)

        # Define the reference vectors for each axis
        x_ref = np.array([1, 0, 0])
        y_ref = np.array([0, 1, 0])
        z_ref = np.array([0, 0, 1])

        # Calculate the angle with respect to each axis
        x_angle = np.abs(np.arccos(np.clip(np.dot(slope, x_ref), -1.0, 1.0)) * 180 / np.pi)
        y_angle = np.abs(np.arccos(np.clip(np.dot(slope, y_ref), -1.0, 1.0)) * 180 / np.pi)
        z_angle = np.abs(np.arccos(np.clip(np.dot(slope, z_ref), -1.0, 1.0)) * 180 / np.pi)

        # Limit the angles to [0, 90] degrees
        # 考虑到ransac计算出来的单位方向向量方向是不确定的，所以要对角度进行处理
        x_angle = min(x_angle, 180 - x_angle)
        y_angle = min(y_angle, 180 - y_angle)
        z_angle = min(z_angle, 180 - z_angle)

        return x_angle, y_angle, z_angle



    def preprocess(self, method, nb_neighbors=100, std_ratio=0.1):
        # Statistic outlier removal(删除与点云的距离比起其他邻域的平均距离远的点)
        # nb_neighbors: 用于计算每个点的邻域的点数
        # std_ratio: 用于计算每个点的邻域的标准差
        # 注意: 对于dense和mittelpoints两种方法处理的点云，要用不同的参数(需要调整)
        # DBSCAN clustering for our mittepunkt
        if method == "RANSAC":
        
            cl, ind = self.open3d_pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors,   # std_ratio越小过滤强度越大
                                                            std_ratio=std_ratio)
            if debug_flag:
                self.display_inlier_outlier(self.open3d_pcd, ind)  #  remove_radius_outlier可视化   
            self.open3d_pcd = self.open3d_pcd.select_by_index(ind)  # removal valid    

            print("pointcloud number(after downsample and removal outlier): ", len(self.open3d_pcd.points))
            
        elif method == "Mittelpoint":

            # # Statistic outlier removal(删除与点云的距离比起其他邻域的平均距离远的点)
            # cl, ind = self.open3d_pcd.remove_statistical_outlier(nb_neighbors=50,   # std_ratio越小过滤强度越大
            #                                                 std_ratio=0.5)
            # if debug_flag:
            #     self.display_inlier_outlier(self.open3d_pcd, ind)  #  remove_radius_outlier可视化   
            # self.open3d_pcd = self.open3d_pcd.select_by_index(ind)  # removal valid    
            # self.open3d_pcd.paint_uniform_color([0,0,1]) # blue

            # DESCAN filter
            pcd_largest_cluster = self.dbscan_filter(self.open3d_pcd)
            

            # visualization
            if debug_flag:
                o3d.visualization.draw_geometries([self.open3d_pcd, pcd_largest_cluster], window_name="DBSCAN cluster(green)", width=800, height=600)

            self.open3d_pcd = pcd_largest_cluster
                
    def dbscan_filter(self, pcd):

        # 
        points = np.asarray(self.open3d_pcd.points)

        # DBSCAN params very important
        # eps defines the distance to neighbours in a cluster
        # min_points defines the minimun number of points required to form a cluster.

        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            labels = np.array(pcd.cluster_dbscan(eps=50, min_points=80, print_progress=True))

        # Find the largest cluster
        max_label = labels.max()
        print(f"point cloud has {max_label + 1} clusters")
        counts = np.bincount(labels[labels != -1])  # Ignore noise labeled as -1
        largest_cluster_label = np.argmax(counts)

        # Extract the points belonging to the largest cluster
        largest_cluster_points = points[labels == largest_cluster_label]        

        # Convert back to Open3D.o3d.geometry.PointCloud
        pcd_largest_cluster = o3d.geometry.PointCloud()
        pcd_largest_cluster.points = o3d.utility.Vector3dVector(largest_cluster_points)
        pcd_largest_cluster.paint_uniform_color([1, 0, 0]) # green

        return pcd_largest_cluster


        
    def voxel2pcd(self, voxel_data):
        # voxel_data Z*Y*X
        index = np.where(voxel_data >= self.thresthold)  
        # pcd = np.vstack((index[0],index[1],index[2]))  # index[0] row vector
        pcd = np.vstack((index[2],index[1],index[0]))  #  zyx -> xyz
        pcd = pcd.T 

        return pcd


    def imf2voxel(self, imf_data):
        # here useless, because the voxel_data is from ITK data
        temp = np.squeeze(np.asarray(imf_data), axis=(0,4))  # z,y,x
        if np.all(temp>=0):
            print("Warning: the voxel data doesnt negative value!")

        return temp
    

    def itk2voxel(self, meta_data):
        # itk image to numpy array
        temp = sitk.GetArrayFromImage(meta_data)   # (154, 418, 449) z,y,x 转换为numpy 

        return temp
    

    def display_inlier_outlier(self, cloud, ind):
        # display inlier and outlier pointcloud
        inlier_cloud = cloud.select_by_index(ind)
        outlier_cloud = cloud.select_by_index(ind, invert=True)
        print("Showing outliers (red) and inliers (gray): ")
        outlier_cloud.paint_uniform_color([1, 0, 0]) # pointcloud color: red
        inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud, origin], window_name="red: outliers, gray: inliers") 


    def create_sphere_at_xyz(self, xyz, colors=None, radius=5, resolution=4):
        """
        create a mesh sphere at xyz
        Args:
            xyz: arr, (3,) numpy array list
            colors: arr, (3, )
        Returns:
            sphere: mesh sphere for visualization
        """
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=resolution)
        # sphere.compute_vertex_normals()
        if colors is None:
            sphere.paint_uniform_color([0,0,0])  # To be changed to the point color.
        else:
            sphere.paint_uniform_color(colors)

        # type check
        if not isinstance(xyz, list):
            xyz = xyz.tolist()
        
    
        sphere = sphere.translate(xyz)

        return sphere
    

    def line_func(self, xyz_0, slope, t):
        """
        create another point along the 3D-line
        Args:
            xyz_0: arr, (3,) interception
            slope: arr, (3, ) slope
            t: length
        Returns:
            another point
        """    
        x = xyz_0[0] + slope[0] * t
        y = xyz_0[1] + slope[1] * t
        z = xyz_0[2] + slope[2] * t

        return (x,y,z)
    
    
    def get_BoundPoints(self, interception, slope, range):
        """
        compute the intersection point of the line and the boundary of the volume
        Args:
            interception: arr, (3,) interception of line
            slope: arr, (3, ) slope of line
            range: arr, (3, ) range of volume z,y,x
        Returns:
            two intersection point arr, (2,3)
        """  
        list = [-150,-100,-50,50,100,150]
        start_point = []
        point_first = None
        point_second = None

        for i in list:
            point = self.line_func(xyz_0=interception, slope=slope, t=i) # 返回元组
            if 0<point[0]<range[0] and 0<point[1]<range[1] and 0<point[2]<range[2]:
                start_point.append(i)
                print("start_point index : ", point)
                break
        
        if len(start_point) == 0:
            raise Exception('no intersection point found!')
        else:
            # print("start_point: ", start_point)
            point_first = self.searchAlongLine(start_point, slope, interception, range, dir="+") # 返回list
            # print("start_point: ", start_point)
            point_second = self.searchAlongLine(start_point, slope, interception, range, dir="-")

        return point_first, point_second
    

    def searchAlongLine(self, start_point, slope, interception, range, dir):
        
        searched_point = None
        start_point = start_point[0]

        while True:
            # print("start_point: ", start_point)
            # print("interception: ", interception)
            # print("slope: ", slope)
            point = self.line_func(xyz_0=interception, slope=slope, t=start_point)

            if int(point[0]) == range[0] or int(point[1]) == range[1] or int(point[2]) == range[2] or \
                int(point[0]) == 0 or int(point[1]) == 0 or int(point[2]) == 0:

                searched_point = list(point)
                print("scurrent index: ", start_point)

                break
                
            if dir == "+":
                start_point += 1
            elif dir == "-":
                start_point -= 1

        return searched_point





            
        

