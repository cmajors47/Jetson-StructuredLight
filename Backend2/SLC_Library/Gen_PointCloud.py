import cv2 as cv
import numpy as np
import Distort_Tools
import Scan_Collecter
import matplotlib.pyplot as plt
import crop
'''
This function takes the inputs from a scan and generates a point cloud
output based 
'''

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------


# def get_xyz(camera1_coords, camera1_M, camera1_R, camera1_T, camera2_coords, camera2_M, camera2_R, camera2_T):
#     # Get the two key equations from camera1
#     camera1_u, camera1_v = camera1_coords
#     # Put the rotation and translation side by side and then multiply with camera matrix
#     camera1_P = camera1_M.dot(np.column_stack((camera1_R,camera1_T)))
#     # Get the two linearly independent equation referenced in the notes
#     camera1_vect1 = camera1_v*camera1_P[2,:]-camera1_P[1,:]
#     camera1_vect2 = camera1_P[0,:] - camera1_u*camera1_P[2,:]
  
#     # Get the two key equations from camera2
#     camera2_u, camera2_v = camera2_coords
#     # Put the rotation and translation side by side and then multiply with camera matrix
#     camera2_P = camera2_M.dot(np.column_stack((camera2_R,camera2_T)))
#     # Get the two linearly independent equation referenced in the notes
#     camera2_vect1 = camera2_v*camera2_P[2,:]-camera2_P[1,:]
#     camera2_vect2 = camera2_P[0,:] - camera2_u*camera2_P[2,:]
  
#     # Stack the 4 rows to create one 4x3 matrix
#     full_matrix = np.row_stack((camera1_vect1, camera1_vect2, camera2_vect1, camera2_vect2))
#     # The first three columns make up A and the last column is b
#     A = full_matrix[:, :3]
#     b = full_matrix[:, 3].reshape((4, 1))
#     # Solve overdetermined system. Note b in the wikipedia article is -b here.
#     # https://en.wikipedia.org/wiki/Overdetermined_system
#     soln = np.linalg.inv(A.T.dot(A)).dot(A.T).dot(-b)
#     return soln
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# def triangulate_point(p1, d1, p2, d2):
#     """
#     Triangulates a 3D point given two rays.

#     Args:
#         p1 (numpy.ndarray): Origin of the first ray (3D vector).
#         d1 (numpy.ndarray): Direction vector of the first ray (3D vector).
#         p2 (numpy.ndarray): Origin of the second ray (3D vector).
#         d2 (numpy.ndarray): Direction vector of the second ray (3D vector).

#     Returns:
#         numpy.ndarray: The triangulated 3D point (3D vector).
#     """

#     # Construct the matrix A
#     A = np.array([d1, -d2]).T

#     # Construct the vector b
#     b = p2 - p1

#     # Solve the system of equations using least squares
#     x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

#     # Calculate the triangulated point
#     point_3d = p1 + x[0] * d1

#     return point_3d

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

corners = crop.find_white_corners("/home/slc/SLC/Backend/procam-calibration/capturesUndistorted/capture_17.png")
K_cam_new = crop.crop_image("/home/slc/SLC/Backend/procam-calibration/capturesUndistorted/capture_*.png", corners)

file = open("/home/slc/SLC/Backend/procam-calibration/Utilities/new_img_dimensions.txt", "r")
cropped_dimensions = file.read().split(", ")
cropped_dimensions = [int(ele) for ele in cropped_dimensions]
file.close()

# Define all camera parameters
proj_w = cropped_dimensions[0] #1024
proj_h = cropped_dimensions[1] #768
cam_w = cropped_dimensions[0] #1024
cam_h = cropped_dimensions[1] #768


black_thr = 4
# Define file locations

graycode = cv.structured_light.GrayCodePattern.create(proj_w, proj_h)
scan_grab = Scan_Collecter.ScanGrabber("/home/slc/SLC/Backend/procam-calibration/Utilities/CapturesCropped")
imgs = scan_grab.GetPhotos()

black_img = imgs.pop()
white_img = imgs.pop()

cam_points = np.empty((cam_w*cam_h, 2), np.uint16)
proj_points = np.empty((cam_w*cam_h, 2), np.uint16)

point_index = 0

# Define calibration parameters
calib_data = Distort_Tools.CalibrationResults("/home/slc/SLC/Backend/procam-calibration/calibration_result.xml")

# Define intrinsic matrices of camera and projector
K_cam = K_cam_new #calib_data.cam_int
K_proj = calib_data.proj_int

# Define extrinsic matrices between camera and projector
rot = calib_data.rotation
trans = calib_data.translation

# Initialize Depth Coordinates
depth = np.zeros((cam_h, cam_w), dtype=np.float32)

xyzPoints = []

# TODO: Liquidate these loops
# Loop through all the points and ignore them if they do not pass the threshold
for x in range(cam_w):
    for y in range(cam_h):
        if int(white_img[y, x] - int(black_img[y,x])) <= black_thr:
            continue
        else:
            # x, y are the camera pixels. point[0], point[1] are the projector pixels.
            # We know parameters
            ret, point = graycode.getProjPixel(imgs, x, y)
            cam_points[point_index] = (x,y)
            proj_points[point_index] = (point[0], point[1]) 
            point_index = point_index + 1

            # 3x3
            #proj_rot = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
            # 3x1
            #proj_trans = np.zeros((3, 1))

            #xyz = get_xyz((x, y), K_cam_new, rot, trans, (point[0], point[1]), K_proj, proj_rot, proj_trans)
            
            #print(xyz)
            xyzPoints.append(xyzPoints)

            print(str(x)+","+str(y))

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

            # projcoords = R*[world coords] + t                 ## {worldcoords=(0,0,0)}
            o_cam_world = trans # Camera Origin in world 
            

            # Set up least-squares problem to find the closest point on the rays
            # Camera origin (0, 0, 0) and projector origin in world coordinates
            o_proj = np.array([0, 0, 0])                # Projector origin in world coordinates
            o_cam = o_cam_world                         # Camera origin in world coordinates
            o_cam = trans.flatten()

            # Define the system for least-squares solution
            # [      ray_cam*ray_cam,       ray_cam*r_cam_world     ]
            # [ -ray_cam*r_cam_world,       r_cam_world*r_cam_world ]
            A = np.array([
                [np.dot(ray_cam, ray_cam), -np.dot(ray_cam, r_cam_world)],
                [-np.dot(ray_cam, r_cam_world), np.dot(r_cam_world, r_cam_world)]
            ])

            # # [ (o_cam-o_proj)*ray_cam,     (o_cam-o_proj)*r_cam_world ]
            b = np.array([
                np.dot(o_cam - o_proj, ray_cam),
                np.dot(o_cam - o_proj, r_cam_world)
            ])
            

            # # Solve for the parameters t and s, which scale the direction vectors
            t, s = np.linalg.solve(A, b)

            # # Find the closest points on each ray
            P_cam = o_cam + t * ray_cam
            P_proj = o_proj + s * r_cam_world
                
            # # Calculate the midpoint between the two closest points as the final 3D point
            P = (P_cam + P_proj) / 2

            # xyz = get_xyz((x,y), np.array(K_cam), np.array(rot), np.array(trans), (point[0], point[1]), np.array(K_proj), proj_rot, proj_trans)
            
            #point = triangulate_point(o_cam_world, r_cam_world, o_proj, ray_proj)
            #print("Triangulated point: ", point, "end")

            depth[y, x] = P[2]#point[2]
            # print(xyz)

#with open("/home/slc/SLC/Backend/procam-calibration/Utilities/3d_img_points.txt", "w") as file:
#    file.write(xyzPoints)

#fig = plt.figure()
#ax = fig.add_subplot(projection='3d')
#for xyz in xyzPoints:
#    ax.scatter(xyz[0], xyz[1], xyz[2])
#plt.savefig('/home/slc/SLC/Backend/procam-calibration/Utilities/pointImg.png', bbox_inches='tight', pad_inches=0, dpi=300)
print(np.array(xyzPoints).shape)

plt.imshow(depth, cmap='gray')
plt.title('Depth Map')
plt.axis('off')  # Turn off axis labels
plt.savefig('/home/slc/SLC/Backend/procam-calibration/Utilities/depth_map.png', bbox_inches='tight', pad_inches=0, dpi=300)


# ------------------------------------------------------------------------------------------------------------------------
# Point Cloud Generation

# Extract valid points for the point cloud
# valid_points = depth > 0  # Mask to filter valid points (non-zero depth)

# Create arrays for X, Y, Z coordinates of valid points

print(depth.shape)

# TODO: depth variable throwing errors
X = np.arange(cam_w)[None, :].repeat(cam_h, axis=0)[depth]
Y = np.arange(cam_h)[:, None].repeat(cam_w, axis=1)[depth]
Z = depth[depth]

# Create a 3D scatter plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X, Z, Y, c=Z, cmap='viridis', marker='.')

# Label the axes
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Depth (Z)")

plt.title("3D Point Cloud from Depth Map")

# Show the plot
plt.savefig('PointCloud.png')


# # Extract x and y coordinates for camera and projector points
# cam_x, cam_y = cam_points[:, 0], cam_points[:, 1]
# proj_x, proj_y = proj_points[:, 0], proj_points[:, 1]

# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.scatter(cam_x, cam_y, s=1, color='blue', label='Camera Points')
# plt.xlabel("X (Camera)")
# plt.ylabel("Y (Camera)")
# plt.title("Camera Pixel Coordinates")
# plt.legend(loc="upper right")

# plt.subplot(1, 2, 2)
# plt.scatter(proj_x, proj_y, s=1, color='red', label='Projector Points')
# plt.xlabel("X (Projector)")
# plt.ylabel("Y (Projector)")
# plt.title("Projector Pixel Coordinates")
# plt.legend(loc="upper right")

# plt.tight_layout()
# plt.savefig("correspondences_plot.png")

# ret, point = graycode.getProjPixel(imgs, 1200, 1000)
# print(point)