import open3d as o3d
import cv2 as cv
import numpy as np

# Load the points from the file
saved_cali = cv.FileStorage("/home/slc/Jetson-StructuredLight/Backend2/3Dpoints.xml", cv.FILE_STORAGE_READ)
points = saved_cali.getNode("a3D_Points").mat()

# Ensure points has the shape (1080, 1920, 3) and dtype (float32)
print("Original points shape:", points.shape)
print("Original points dtype:", points.dtype)

# Reshape points to (height * width, 3) = (1080 * 1920, 3)
points_reshaped = points.reshape((-1, 3))
points_reshaped = points_reshaped[::100]
# Convert to float64 for Open3D
#points_reshaped = points_reshaped.astype(np.float64)

# Remove any rows with NaN or Inf values
valid_points = points_reshaped[np.isfinite(points_reshaped).all(axis=1)]

# Convert the valid points to an Open3D PointCloud
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(valid_points)
cl, ind = point_cloud.remove_statistical_outlier(nb_neighbors=50, std_ratio=2.0)
filtered_pcd = point_cloud.select_by_index(ind)

# Visualize the point cloud
o3d.io.write_point_cloud("output_point_cloud.ply", filtered_pcd)
#o3d.visualization.draw_geometries([point_cloud])