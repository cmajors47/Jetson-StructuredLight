import cv2 as cv
import numpy as np
from Distort_Tools import CalibrationResults
from Scan_Collecter import ScanGrabber
import matplotlib.pyplot as plt
'''
This function takes the inputs from a scan and generates a point cloud
output based 
'''

# Define all camera parameters
proj_w = 1280
proj_h = 720    
cam_w = 1920
cam_h = 1080

black_thr = 4
# Define file locations

graycode = cv.structured_light.GrayCodePattern.create(proj_w, proj_h)
scan_grab = ScanGrabber()
imgs = scan_grab.GetPhotos()

black_img = imgs.pop()
white_img = imgs.pop()

cam_points = np.empty((cam_w*cam_h, 2), np.uint16)
proj_points = np.empty((cam_w*cam_h, 2), np.uint16)

point_index = 0

# Define calibration parameters
calib_data = CalibrationResults("calibration_result.xml")

# Define intrinsic matrices of camera and projector
K_cam = calib_data.cam_int
K_proj = calib_data.proj_int

# Define extrinsic matrices between camera and projector
rot = calib_data.rotation
trans = calib_data.translation

# Initialize Depth Coordinates
depth = np.zeros((cam_h, cam_w), dtype=np.float32)

# Loop through all the points and ignore them if they do not pass the threshold
for x in range(cam_w):
    for y in range(cam_h):
        if int(white_img[y, x] - int(black_img[y,x])) <= black_thr:
            continue
        else:
                ret, point = graycode.getProjPixel(imgs, x, y)
                cam_points[point_index] = (x,y)
                proj_points[point_index] = (point[0], point[1]) 
                point_index = point_index + 1
                
                # Generate 3D rays for camera and projector in homogeneous coordinates
                cam_homogeneous = np.array([x, y, 1])  # Camera pixel (x, y, 1)
                proj_homogeneous = np.array([point[0], point[1], 1])  # Projector pixel (u, v, 1)
            
                # Use the intrinsic matrices to calculate the 3D direction vectors
                ray_cam = np.linalg.inv(K_cam) @ cam_homogeneous
                ray_proj = np.linalg.inv(K_proj) @ proj_homogeneous

                # Normalize the rays to make them unit vectors
                ray_cam /= np.linalg.norm(ray_cam)
                ray_proj /= np.linalg.norm(ray_proj)
                
                # Convert the camera rays from its local coordinate system to the projector's (world coordinates)
                r_cam_world = rot @ ray_cam # Direction
                o_cam_world = trans # Projector Origin in world
                
                # Set up least-squares problem to find the closest point on the rays
                # Camera origin (0, 0, 0) and projector origin in world coordinates
                o_proj = np.array([0, 0, 0])                 # Camera origin in world coordinates
                o_cam = o_cam_world        # Projector origin in world coordinates
                o_cam = trans.flatten()

                # Define the system for least-squares solution
                A = np.array([
                    [np.dot(ray_cam, ray_cam), -np.dot(ray_cam, r_cam_world)],
                    [-np.dot(ray_cam, r_cam_world), np.dot(r_cam_world, r_cam_world)]
                ])

                b = np.array([
                    np.dot(o_cam - o_proj, ray_cam),
                    np.dot(o_cam - o_proj, r_cam_world)
                ])
                # Solve for the parameters t and s, which scale the direction vectors
                t, s = np.linalg.solve(A, b)

                # Find the closest points on each ray
                P_cam = o_cam + t * ray_cam
                P_proj = o_proj + s * r_cam_world

                # Calculate the midpoint between the two closest points as the final 3D point
                P = (P_cam + P_proj) / 2
                
                depth[y, x] = P[2] 
                   
plt.imshow(depth, cmap='gray')
plt.colorbar(label='Depth (mm)')
plt.title('Depth Map')
plt.axis('off')  # Turn off axis labels
plt.savefig('depth_map.png', bbox_inches='tight', pad_inches=0, dpi=300)
