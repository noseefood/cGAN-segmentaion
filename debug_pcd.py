import open3d as o3d
import numpy as np


def display_inlier_outlier(cloud, ind):
    # display inlier and outlier pointcloud
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)
    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0]) # pointcloud color: red
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud, origin], window_name="red: outliers, gray: inliers") 


# Read the point cloud
pcd = o3d.io.read_point_cloud("./results/pointcloud.pcd")


# pcd = o3d.geometry.PointCloud.voxel_down_sample(pcd, voxel_size=2)

# Visualize the point cloud
pcd.paint_uniform_color([1, 0, 0])

# statistical outlier removal
cl, ind = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=0.5)

# radius outlier removal
# cl, ind = pcd.remove_radius_outlier(nb_points=50, radius=30)


display_inlier_outlier(pcd, ind)

pcd = pcd.select_by_index(ind) 


# DCSCAN
# DESCAN filter
# Convert Open3D.o3d.geometry.PointCloud to numpy array
points = np.asarray(pcd.points)

# DBSCAN
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
pcd_largest_cluster.paint_uniform_color([0, 1, 0])


o3d.visualization.draw_geometries([pcd, pcd_largest_cluster])

# sati