import cv2 as cv
import numpy as np
from Distort_Tools import CalibrationResults
from Scan_Collecter import ScanGrabber
import matplotlib.pyplot as plt
import open3d as o3d
'''
This function takes the inputs from a scan and generates a point cloud
output based 
'''

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
depth = np.zeros((cam_h, cam_w,3), dtype=np.float64)

# note with the shadow mask: Should adaptive thresholding be used? This might nee dto be purseud if too many pixels are hidden
shadow_mask = cv.threshold(abs(white_img - black_img), black_thr, 1, cv.THRESH_BINARY)[1]
point_cloud = np.zeros((cam_h, cam_w, 3), dtype=np.float64)
# Loop through all the points and ignore them if they do not pass the threshold
for x in range(cam_w):
  for y in range(cam_h):
    if shadow_mask[y, x] == 1:
      ret, point = graycode.getProjPixel(imgs, x, y)
      cam_points[point_index] = (x, y)
      proj_points[point_index] = (point[0], point[1]) 
      
      # Generate 3D rays for camera and projector in homogeneous coordinates
      cam_homog = np.array([x, y, 1])  # Camera pixel (x, y, 1)
      proj_homog = np.array([point[0], point[1], 1])  # Projector pixel (u, v, 1)
  
      # Use the intrinsic matrices to calculate the 3D direction vectors
      ray_cam = np.linalg.inv(K_cam) @ cam_homog
      ray_proj = np.linalg.inv(K_proj) @ proj_homog
      # Normalize the rays to make them unit vectors
      ray_cam /= np.linalg.norm(ray_cam)
      ray_proj /= np.linalg.norm(ray_proj)
      
      # Convert the camera rays from its local coordinate system to the projector's (world coordinates)
      r_cam_world = rot @ ray_cam # Direction
      o_cam_world = trans # Projector Origin in world
      
      # Set up least-squares problem to find the closest point on the rays
      # Camera origin (0, 0, 0) and projector origin in world coordinates
      o_proj = np.array([20, 5, 13])                 # Camera origin in world coordinates
      o_cam = o_cam_world        # Projector origin in world coordinates
      o_cam = trans.flatten()
      # Define the system for least-squares solution
      A = np.array([
          [np.dot(ray_cam, ray_cam), -np.dot(ray_cam, r_cam_world)],
          [-np.dot(ray_cam, r_cam_world), np.dot(r_cam_world, r_cam_world)]
      ])
      b = np.array([
          np.dot(o_proj - o_cam, ray_cam),
          np.dot(o_proj - o_cam, r_cam_world)
      ])
      # Solve for the parameters t and s, which scale the direction vectors
      t, s = np.linalg.solve(A, b)
      # Find the closest points on each ray
      P_cam = o_cam + t * ray_cam
      P_proj = o_proj + s * r_cam_world
      # Calculate the midpoint between the two closest points as the final 3D point
      P = (P_cam + P_proj) / 2
      P[2] = 255-P[2]
      depth[y, x] =P[2]
      point_cloud[y,x]=P
    point_index = point_index + 1

depth_min, depth_max = np.min(depth), np.max(depth)
depth_normalized = ((depth-depth_min)/(depth_max-depth_min))*255
depth_normalized = np.uint8(depth_normalized)
fs = cv.FileStorage('3Dpoints.xml', cv.FILE_STORAGE_WRITE)
fs.write('a3D_Points', point_cloud)
plt.imshow(depth_normalized, cmap='gray')
plt.colorbar(label='Depth (mm)')
plt.title('Depth Map')
plt.axis('off')  # Turn off axis labels
plt.savefig('depth_map.png', bbox_inches='tight', pad_inches=0, dpi=300)

# X = np.arange(cam_w)[None, :].repeat(cam_h, axis=0)[depth.astype(np.uint16)]
# Y = np.arange(cam_h)[:, None].repeat(cam_w, axis=1)[depth.astype(np.uint16)]
# Z = depth[depth.astype(np.uint16)]

# print(np.stack(X,Y,Z))


# # Create a 3D scatter plot
# fig = plt.figure(figsize=(10, 7))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(X, Z, Y, c=Z, cmap='viridis', marker='.')

# # Label the axes
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Depth (Z)")

# plt.title("3D Point Cloud from Depth Map")

# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(Parray)
# o3d.visualization.draw_geometries([pcd])