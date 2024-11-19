import cv2 as cv
import numpy as np
from Distort_Tools import CalibrationResults
from Scan_Collecter import ScanGrabber
import matplotlib.pyplot as plt
import open3d as o3d

# Define all camera parameters
proj_w = 1280
proj_h = 720
cam_w = 1920
cam_h = 1080

black_thr = 40

# Define file locations
graycode = cv.structured_light.GrayCodePattern.create(proj_w, proj_h)
scan_grab = ScanGrabber("./captureUndistortedOptim")
imgs = scan_grab.GetPhotos()

black_img = imgs.pop()
white_img = imgs.pop()

cam_points = np.empty((cam_w*cam_h, 2), np.uint16)
proj_points = np.empty((cam_w*cam_h, 2), np.uint16)

point_index = 0

# Define calibration parameters
calib_data = CalibrationResults("/home/slc/Jetson-StructuredLight/Backend2/calibration_result.xml")

# Define intrinsic matrices of camera and projector
K_cam = calib_data.cam_int
K_proj = calib_data.proj_int

# Define extrinsic matrices between camera and projector
rot = calib_data.rotation
trans = calib_data.translation

# Initialize Depth Coordinates
depth = np.zeros((cam_h, cam_w, 3), dtype=np.float64)
shadow_mask = cv.threshold(abs(white_img - black_img), black_thr, 1, cv.THRESH_BINARY)[1]

point_cloud = []

# Loop through all the points and ignore them if they do not pass the threshold
for x in range(cam_w):
    for y in range(cam_h):
        if shadow_mask[y, x] == 1:
            ret, point = graycode.getProjPixel(imgs, x, y)
            cam_points[point_index] = (x, y)
            proj_points[point_index] = (point[0], point[1]) 

            # Store corresponding points for triangulation
            cam_pixel = np.array([x, y, 1.0])  # Camera pixel (x, y, 1)
            proj_pixel = np.array([point[0], point[1], 1.0])  # Projector pixel (u, v, 1)

            # Add points to lists
            cam_points_list = cam_points[point_index]
            proj_points_list = proj_points[point_index]

            point_index += 1

# Convert to homogeneous coordinates for triangulation (2D homogeneous coordinates)
cam_points_homog = np.hstack([cam_points, np.ones((cam_points.shape[0], 1))])  # (N, 3) -> (x, y, 1)
proj_points_homog = np.hstack([proj_points, np.ones((proj_points.shape[0], 1))])  # (N, 3) -> (u, v, 1)

# Transpose to get the shape (2, N) as required by cv2.triangulatePoints()
cam_points_homog = cam_points_homog[:, :2].T  # (2, N)
proj_points_homog = proj_points_homog[:, :2].T  # (2, N)

# Assuming you already have cam_points, proj_points, and the calibration matrices

# Create 3x4 projection matrices for both the camera and the projector

# Assuming you already have cam_points, proj_points, and the calibration matrices

# Create 3x4 projection matrices for both the camera and the projector
P_cam = K_cam @ np.hstack([np.eye(3), np.zeros((3, 1))])  # Camera projection matrix (3x4)
P_proj = K_proj @ np.hstack([rot, trans.reshape(-1, 1)])  # Projector projection matrix (3x4)

# Convert points to homogeneous coordinates by adding a third coordinate (1)
cam_points_homog = np.hstack([cam_points, np.ones((cam_points.shape[0], 1))])  # (N, 3) -> (N, 3, 1)
proj_points_homog = np.hstack([proj_points, np.ones((proj_points.shape[0], 1))])  # (N, 3) -> (N, 3, 1)

# Convert to the correct shape (2, N) for triangulation
cam_points_homog = cam_points_homog[:, :2].T  # (2, N)
proj_points_homog = proj_points_homog[:, :2].T  # (2, N)

# Define an empty array to store the result
points_3d_homog = np.empty((4, cam_points_homog.shape[1]), dtype=np.float32)

# Triangulate points using cv2.triangulatePoints()
points_3d_homog = cv.triangulatePoints(P_cam, P_proj, cam_points_homog, proj_points_homog)

# Now points_3d_homog contains the 4D homogeneous coordinates
# Validate W and normalize homogeneous coordinates to get Euclidean 3D points (divide by W)
W = points_3d_homog[3, :]  # Extract W (the 4th row)

# Check for points where W is zero or very small
valid_points_mask = np.abs(W) > 1e-6  # Threshold to avoid division by zero or near-zero values

# Initialize 3D points with NaN (or other values) for invalid points
points_3d = np.full_like(points_3d_homog[:3], np.nan)

# Only normalize points where W is valid (not zero or near-zero)
points_3d[:, valid_points_mask] = points_3d_homog[:3, valid_points_mask] / W[valid_points_mask]

# Now points_3d contains the 3D coordinates in Euclidean space (3xN)
point_cloud = points_3d.T  # Convert to Nx3 format
#depth = point_cloud[2]
# Create a point cloud object with Open3D
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(point_cloud)

# # Visualize the point cloud
# o3d.visualization.draw_geometries([pcd])

# # Optionally, save the point cloud to a file
# o3d.io.write_point_cloud("output_point_cloud.ply", pcd)

# Save depth map (as an image or data)
depth_min, depth_max = np.min(depth), np.max(depth)
if depth_max > depth_min:
    depth_normalized = ((depth - depth_min) / (depth_max - depth_min)) * 255
else:
    depth_normalized = np.zeros_like(depth, dtype=np.uint8)

depth_normalized = np.uint8(depth_normalized)  # Ensure the depth map is in uint8 format

fs = cv.FileStorage('3Dpoints.xml', cv.FILE_STORAGE_WRITE)
fs.write('a3D_Points', point_cloud)

# Show the depth map
import matplotlib.pyplot as plt
plt.imshow(depth_normalized, cmap='gray')
plt.colorbar(label='Depth (mm)')
plt.title('Depth Map')
plt.axis('off')  # Turn off axis labels
plt.savefig('depth_map.png', bbox_inches='tight', pad_inches=0, dpi=300)


