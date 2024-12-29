# coding: UTF-8

import os
import os.path
import glob
import argparse
import cv2 as cv
import numpy as np
import json
import shutil
from typing import List
import open3d as o3d


def calibrate_main(proj_height,proj_width,chess_vert,chess_hori,chess_block_size,graycode_step):

    proj_shape = (proj_height, proj_width)
    chess_shape = (chess_vert, chess_hori)
    chess_block_size = chess_block_size
    gc_step = graycode_step
    black_thr = 40
    white_thr = 5

    camera_param_file = str()

    dirnames = sorted(glob.glob('./capture_*'))
    if len(dirnames) == 0:
        print('Directories \'./capture_*\' were not found')
        return

    print('Searching input files ...')
    used_dirnames = []
    gc_fname_lists = []
    for dname in dirnames:
        gc_fnames = sorted(glob.glob(dname + '/graycode_*'))
        if len(gc_fnames) == 0:
            continue
        used_dirnames.append(dname)
        gc_fname_lists.append(gc_fnames)
        print(' \'' + dname + '\' was found')

    camP = None
    cam_dist = None
    path, ext = os.path.splitext(camera_param_file)
    if(ext == ".json"):
        camP,cam_dist = loadCameraParam(camera_param_file)
        print('load camera parameters')
        print(camP)
        print(cam_dist)

    calibrate(used_dirnames, gc_fname_lists,
            proj_shape, chess_shape, chess_block_size, gc_step, black_thr, white_thr,
            camP, cam_dist)


def printNumpyWithIndent(tar, indentchar):
    print(indentchar + str(tar).replace('\n', '\n' + indentchar))

def loadCameraParam(json_file):
    with open(json_file, 'r') as f:
        param_data = json.load(f)
        P = param_data['camera']['P']
        d = param_data['camera']['distortion']
        return np.array(P).reshape([3,3]), np.array(d)

def calibrate(dirnames, gc_fname_lists, proj_shape, chess_shape, chess_block_size, gc_step, black_thr, white_thr, camP, camD):
    objps = np.zeros((chess_shape[0]*chess_shape[1], 3), np.float32)
    objps[:, :2] = chess_block_size * \
        np.mgrid[0:chess_shape[0], 0:chess_shape[1]].T.reshape(-1, 2)

    print('Calibrating ...')
    gc_height = int((proj_shape[0]-1)/gc_step)+1
    gc_width = int((proj_shape[1]-1)/gc_step)+1
    graycode = cv.structured_light_GrayCodePattern.create(
        gc_width, gc_height)
    graycode.setBlackThreshold(black_thr)
    graycode.setWhiteThreshold(white_thr)

    cam_shape = cv.imread(gc_fname_lists[0][0], cv.IMREAD_GRAYSCALE).shape
    patch_size_half = int(np.ceil(cam_shape[1] / 180))
    print('  patch size :', patch_size_half * 2 + 1)

    cam_corners_list = []
    cam_objps_list = []
    cam_corners_list2 = []
    proj_objps_list = []
    proj_corners_list = []
    for dname, gc_filenames in zip(dirnames, gc_fname_lists):
        print('  checking \'' + dname + '\'')
        if len(gc_filenames) != graycode.getNumberOfPatternImages() + 2:
            print('Error : invalid number of images in \'' + dname + '\'')
            return None

        imgs = []
        for fname in gc_filenames:
            img = cv.imread(fname, cv.IMREAD_GRAYSCALE)
            if cam_shape != img.shape:
                print('Error : image size of \'' + fname + '\' is mismatch')
                return None
            imgs.append(img)
        black_img = imgs.pop()
        white_img = imgs.pop()

        res, cam_corners = cv.findChessboardCorners(white_img, chess_shape)
        if not res:
            print('Error : chessboard was not found in \'' +
                gc_filenames[-2] + '\'')
            return None
        cam_objps_list.append(objps)
        cam_corners_list.append(cam_corners)

        proj_objps = []
        proj_corners = []
        cam_corners2 = []
        # viz_proj_points = np.zeros(proj_shape, np.uint8)
        for corner, objp in zip(cam_corners, objps):
            c_x = int(round(corner[0][0]))
            c_y = int(round(corner[0][1]))
            src_points = []
            dst_points = []
            for dx in range(-patch_size_half, patch_size_half + 1):
                for dy in range(-patch_size_half, patch_size_half + 1):
                    x = c_x + dx
                    y = c_y + dy
                    if int(white_img[y, x]) - int(black_img[y, x]) <= black_thr:
                        continue
                    err, proj_pix = graycode.getProjPixel(imgs, x, y)
                    if not err:
                        src_points.append((x, y))
                        dst_points.append(gc_step*np.array(proj_pix))
            if len(src_points) < patch_size_half**2:
                print(
                    '    Warning : corner', c_x, c_y,
                    'was skiped because decoded pixels were too few (check your images and threasholds)')
                continue
            h_mat, inliers = cv.findHomography(
                np.array(src_points), np.array(dst_points))
            point = h_mat@np.array([corner[0][0], corner[0][1], 1]).transpose()
            point_pix = point[0:2]/point[2]
            proj_objps.append(objp)
            proj_corners.append([point_pix])
            cam_corners2.append(corner)
            # viz_proj_points[int(round(point_pix[1])),
            #                 int(round(point_pix[0]))] = 255
        if len(proj_corners) < 3:
            print('Error : too few corners were found in \'' +
                dname + '\' (less than 3)')
            return None
        proj_objps_list.append(np.float32(proj_objps))
        proj_corners_list.append(np.float32(proj_corners))
        cam_corners_list2.append(np.float32(cam_corners2))
        # cv2.imwrite('visualize_corners_projector_' +
        #             str(cnt) + '.png', viz_proj_points)
        # cnt += 1

    print('Initial solution of camera\'s intrinsic parameters')
    cam_rvecs = []
    cam_tvecs = []
    if(camP is None):
        ret, cam_int, cam_dist, cam_rvecs, cam_tvecs = cv.calibrateCamera(
            cam_objps_list, cam_corners_list, cam_shape, None, None, None, None)
        print('  RMS :', ret)
    else:
        for objp, corners in zip(cam_objps_list, cam_corners_list):
            ret, cam_rvec, cam_tvec = cv.solvePnP(objp, corners, camP, camD) 
            cam_rvecs.append(cam_rvec)
            cam_tvecs.append(cam_tvec)
            print('  RMS :', ret)
        cam_int = camP
        cam_dist = camD
    print('  Intrinsic parameters :')
    printNumpyWithIndent(cam_int, '    ')
    print('  Distortion parameters :')
    printNumpyWithIndent(cam_dist, '    ')
    print()

    print('Initial solution of projector\'s parameters')
    ret, proj_int, proj_dist, proj_rvecs, proj_tvecs = cv.calibrateCamera(
        proj_objps_list, proj_corners_list, proj_shape, None, None, None, None)
    print('  RMS :', ret)
    print('  Intrinsic parameters :')
    printNumpyWithIndent(proj_int, '    ')
    print('  Distortion parameters :')
    printNumpyWithIndent(proj_dist, '    ')
    print()

    print('=== Result ===')
    ret, cam_int, cam_dist, proj_int, proj_dist, cam_proj_rmat, cam_proj_tvec, E, F = cv.stereoCalibrate(
        proj_objps_list, cam_corners_list2, proj_corners_list, cam_int, cam_dist, proj_int, proj_dist, None)
    print('  RMS :', ret)
    print('  Camera intrinsic parameters :')
    printNumpyWithIndent(cam_int, '    ')
    print('  Camera distortion parameters :')
    printNumpyWithIndent(cam_dist, '    ')
    print('  Projector intrinsic parameters :')
    printNumpyWithIndent(proj_int, '    ')
    print('  Projector distortion parameters :')
    printNumpyWithIndent(proj_dist, '    ')
    print('  Rotation matrix / translation vector from camera to projector')
    print('  (they translate points from camera coord to projector coord) :')
    printNumpyWithIndent(cam_proj_rmat, '    ')
    printNumpyWithIndent(cam_proj_tvec, '    ')
    print()

    fs = cv.FileStorage('calibration_result.xml', cv.FILE_STORAGE_WRITE)
    fs.write('img_shape', cam_shape)
    fs.write('rms', ret)
    fs.write('cam_int', cam_int)
    fs.write('cam_dist', cam_dist)
    fs.write('proj_int', proj_int)
    fs.write('proj_dist', proj_dist)
    fs.write('rotation', cam_proj_rmat)
    fs.write('translation', cam_proj_tvec)
    fs.release()

### CAP SEQUENCE ###
def CalProjectAndCapture(camera,proj_img, pattern_count, target_dir):
    # Display proj_img and wait for the screen to update
    # 45 ms should work for a 30fps projector, though the latency is unknown - come back to this perhaps?
    cv.imshow("main", proj_img)
    cv.waitKey(300)
    # Get photo and check if it was received. Do it twice to flush out the input
    ret, img = camera.read()
    ret, img = camera.read()
    if not ret:
        camera.release()
        raise Exception("Failed to take a photo with the camera.")
    
    cv.imwrite(f"{target_dir}/graycode_{pattern_count:02d}.png", img)
    return

def cap_sequence_main():
    # Set max captures to take
    num_captures = 5

    # Initialize the camera and set camera properties
    camera = cv.VideoCapture(-1)
    camera_frame = (1920, 1080) #x,y dimensions (same as projector)
    camera.set(cv.CAP_PROP_FRAME_WIDTH, camera_frame[0])
    camera.set(cv.CAP_PROP_FRAME_HEIGHT, camera_frame[1]) 
    camera.set(cv.CAP_PROP_EXPOSURE, 0)
    # Note that the internal buffer will not be updated. Two reads need to be done to get one photo - the first to clear the buffer, the second
    # to get the most recent input. Check Link: https://www.reddit.com/r/opencv/comments/p415cc/question_how_do_i_get_a_fresh_frame_taken_after/
    camera.set(cv.CAP_PROP_BUFFERSIZE, 1)

    # Setup a window to be used
    cv.namedWindow("main", cv.WINDOW_NORMAL)
    cv.setWindowProperty("main", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

    # Find all needed files
    graycode_names = sorted(glob.glob("graycode_pattern/pattern_*.png"))
    capture_folders = glob.glob("capture_*/")
    # Remove directories with same names
    for folder_name in capture_folders:
        shutil.rmtree(folder_name)
    # Construct target directories
    target_dirs = [f"capture_{c}" for c in range(0, num_captures)]
    for file in target_dirs:
        os.mkdir(file)

    # key: file name, value: image file data
    projection_patterns = {f_name: cv.imread(f_name) for f_name in graycode_names}

    # Repeat process for each capture
    for i, dir in enumerate(target_dirs):
        [CalProjectAndCapture(camera,projection_patterns[file_name], j, dir) for j, file_name in enumerate(projection_patterns)]
        print(f"There are {num_captures - i} remaining capture(s). Adjust target and press any key to continue.")
        cv.waitKey(0)

    # Release the camera
    camera.release()


class CalibrationResults:
    def __init__(self, file_path: str ="calibration_result.xml"):
        # Begin by opening the file. If it worked, proceed.
        saved_cali = cv.FileStorage(file_path, cv.FILE_STORAGE_READ)
        if not saved_cali.isOpened():
            raise Exception(f"Opening the file '{file_path}' from directory '{os.getcwd()}' caused an issue.")
        
        # Write each of the important constants from the file, then close it
        self.img_shape = saved_cali.getNode("img_shape").mat().flatten()
        self.cam_int = saved_cali.getNode("cam_int").mat()
        self.cam_dist = saved_cali.getNode("cam_dist").mat()
        self.proj_int = saved_cali.getNode("proj_int").mat()
        self.proj_dist = saved_cali.getNode("proj_dist").mat()
        self.rotation = saved_cali.getNode("rotation").mat()
        self.translation = saved_cali.getNode("translation").mat()
        
        saved_cali.release()
        
        # The new_camera_matrix shennanigans could be done but it is not necessary.
        
    # This function uses the saved parameters to return an undistorted image
    def undistort_camera_img(self, input_image: cv.Mat) -> cv.Mat:
        result = cv.undistort(input_image, self.cam_int, self.cam_dist, None, self.cam_int)
        return result
    
    # This function uses the saved parameters to return an undistorted projector image
    def undistort_proj_img(self, input_image: cv.Mat) -> cv.Mat:
        result = cv.undistort(input_image, self.proj_int, self.proj_dist, None, self.proj_int)
        return result


class ScanGrabber:
    # Grabs all of the images following the given naming convention. Initialize sets these variables.
    def __init__(self, target_directory: str = "./captureUndistortedOptim", lnx_photo_search: str = "capture_*.png"):
        # Define variables to be used in other functions
        self.target_directory = target_directory
        self.lnx_photo_search = lnx_photo_search
        self.photo_count = 0
        
    # Returns a list of photo names to give to the user in Gray Scale
    def GetPhotos(self) -> List[cv.Mat]:
        # Find the photo names, note that it is in alphabetical order due to sorted
        photo_names = sorted(glob.glob(self.target_directory + "/" + self.lnx_photo_search))
        if len(photo_names) == 0:
            raise Exception(f"Failed to find any photos with the following search path: {self.target_directory + self.lnx_photo_search}")
        
        # Start adding images to the final array
        final: list[cv.Mat] = []
        for name in photo_names:
            # Add in photos
            img = cv.imread(name, cv.IMREAD_GRAYSCALE)
            final.append(img)
        
        return final
    
    def GetColorImg(self) -> cv.Mat:
        """Gets the Fully illuminated image in full color

        Raises:
            Exception: failed to find the proper search path

        Returns:
            cv.Mat: The white image in RGB format
        """
        # Find the white photo in color
        # Find the photo names, note that it is in alphabetical order due to sorted
        photo_names = sorted(glob.glob(self.target_directory + "/" + self.lnx_photo_search))
        if len(photo_names) == 0:
            raise Exception(f"Failed to find any photos with the following search path: {self.target_directory + self.lnx_photo_search}")

        white_img = cv.imread(photo_names[-2])
        white_img = cv.cvtColor(white_img, cv.COLOR_BGR2RGB)
        
        return white_img


def GenPointCloud():
    # Define all parameters, this should be changed later to accept variable input from ScanMain.py
    proj_w = 1280
    proj_h = 720
    cam_w = 1920
    cam_h = 1080

    black_thr = 40
    white_thr = 5

    # Setup graycode object for pixel correspondences
    graycode = cv.structured_light.GrayCodePattern.create(proj_w, proj_h)
    graycode.setBlackThreshold(black_thr)
    graycode.setWhiteThreshold(white_thr)

    # Define file locations
    scan_grab = ScanGrabber("./ScanCaptures")
    imgs = scan_grab.GetPhotos()
    colored_img = scan_grab.GetColorImg()

    black_img = imgs.pop()
    white_img = imgs.pop()

    cam_points = np.empty((cam_w*cam_h, 2), np.uint16)
    proj_points = np.empty((cam_w*cam_h, 2), np.uint16)
    colors = np.empty((cam_w*cam_h, 3), np.uint8)

    point_index = 0

    # Grab the calibration results
    calib_data = CalibrationResults("calibration_result.xml")

    K_cam = calib_data.cam_int
    K_proj = calib_data.proj_int

    rot = calib_data.rotation
    trans = calib_data.translation

    # Create shadow mask
    shadow_mask = cv.threshold(abs(white_img - black_img), black_thr, 1, cv.THRESH_BINARY)[1]

    point_cloud = []

    # Loop through all the points and ignore them if they do not pass the threshold
    for x in range(cam_w):
        for y in range(cam_h):
            if shadow_mask[y, x] == 1:
                ret, point = graycode.getProjPixel(imgs, x, y)
                
                if not ret:
                    cam_points[point_index] = (x, y)
                    proj_points[point_index] = (point[0], point[1]) 

                    # Add point color to the list
                    colors[point_index, :] = colored_img[y,x, :]

                    point_index += 1


    # Create 3x4 projection matrices for both the camera and the projector
    P_cam = K_cam @ np.hstack([np.eye(3), np.zeros((3, 1))])  # Camera projection matrix (3x4)
    P_proj = K_proj @ np.hstack([rot, trans.reshape(-1, 1)])  # Projector projection matrix (3x4)

    # Convert points to homogeneous coordinates by adding a third coordinate (1)
    cam_points_homog = np.hstack([cam_points, np.ones((cam_points.shape[0], 1))]) 
    proj_points_homog = np.hstack([proj_points, np.ones((proj_points.shape[0], 1))])

    # Convert to the correct shape (2, N) for triangulation
    cam_points_homog = cam_points_homog[:, :2].T  
    proj_points_homog = proj_points_homog[:, :2].T 

    # Triangulate points
    points_3d_homog = cv.triangulatePoints(P_cam, P_proj, cam_points_homog, proj_points_homog)

    # Validate W and normalize homogeneous coordinates to get 3D points (divide by W)
    W = points_3d_homog[3, :]  # Extract W (the 4th row)

    # Check for points where W is zero or very small
    valid_points_mask = np.abs(W) > 1e-6  # Threshold to avoid division by zero or near-zero values

    # Initialize 3D points with NaN (or other values) for invalid points
    points_3d = np.full_like(points_3d_homog[:3], np.nan)

    # Only normalize points where W is valid (not zero or near-zero)
    points_3d[:, valid_points_mask] = points_3d_homog[:3, valid_points_mask] / W[valid_points_mask]
    point_cloud = points_3d.T  # Convert to Nx3 format
    
    # Normalize Colors
    colors = colors[0:point_index, :].astype(np.float32) / 255

    fs = cv.FileStorage('3Dpoints.xml', cv.FILE_STORAGE_WRITE)
    fs.write('a3D_Points', point_cloud[0:point_index, :])
    fs.write('Colors', colors)



### GenGrayCodeImgs ###
     

def constructImage(height, width, pattern, step):
    img = np.zeros((height, width), np.uint8)
    # List comprehension is faster than for loop
    img[:] = [[pattern[int(y/step), int(x/step)] for x in range(width)] for y in range(height)]
    return img

def GenGraycodeImgs_main(proj_height,proj_width,graycode_step):
    TARGET_DIR = './graycode_pattern'
    CAPTURED_DIR = './capture_*'
    step = graycode_step
    height = proj_height
    width = proj_width
    gc_height = int((height - 1 ) / step) + 1
    gc_width = int((width - 1) / step) + 1
    
    graycode = cv.structured_light.GrayCodePattern.create(gc_width, gc_height)
    patterns = graycode.generate()[1]
    
    # Expand image size
    exp_patterns = [constructImage(height, width, pat, step) for pat in patterns]
    # Append white and black patterns
    exp_patterns.append(np.full((height, width), fill_value=255, dtype=np.uint8))
    exp_patterns.append(np.zeros((height, width), dtype=np.uint8))

    if not os.path.exists(TARGET_DIR):
        os.mkdir(TARGET_DIR)

    for i, pat in enumerate(exp_patterns):
        #cv2.imwrite(TARGET_DIR + '/pattern_' + str(i).zfill(2) + '.png', pat)
        cv.imwrite(f"{TARGET_DIR}/pattern_{str(i).zfill(2)}.png", pat)

    output = (
        f"=== Result ===\n"
        f"\'{TARGET_DIR}/pattern_00.png ~ pattern_{str(len(exp_patterns)-1)}.png\' were generated.\n\n"
        f"=== Next step ===\n"
        f"Project patterns and save captured images as \'{CAPTURED_DIR}/graycode_*.png\'\n\n"
        f"    ./ --- capture_1/ --- graycode_00.png\n"
        f"        |              |- graycode_01.png\n"
        f"        |              |        .\n"
        f"        |              |        .\n"
        f"        |              |- graycode_{str(len(exp_patterns)-1)}.png\n"
        f"        |- capture_2/ --- graycode_00.png\n"
        f"        |              |- graycode_01.png\n"
        f"        |      .       |        .\n"
        f"        |      .       |        .\n\n"
        f"It is recommended to capture more than 5 times.\n"
    )
    print(output)


### Scan ###

def ScanProjectAndCapture(camera, proj_img, pattern_count, target_dir):
    # Display proj_img and wait for the screen to update
    # 45 ms should work for a 30fps projector, though the latency is unknown - come back to this perhaps?
    cv.imshow("main", proj_img)
    cv.waitKey(300)
    # Get photo and check if it was received. Do it twice to flush out the input
    ret, img = camera.read()
    ret, img = camera.read()
    if not ret:
        camera.release()
        raise Exception("Failed to take a photo with the camera.")
    
    cv.imwrite(f"{target_dir}/capture_{pattern_count:02d}.png", img)
    print(f"SAVED: {target_dir}/capture_{pattern_count:02d}.png")
    return

def Scan_main(CAMERA_WIDTH, CAMERA_HEIGHT):

    os.system("export DISPLAY=:0")

    # Get command line arguments
    GRAYCODE_DIRECTORY = "./graycode_pattern"

    if not os.path.isdir(GRAYCODE_DIRECTORY):
        raise Exception("ERROR: Gray Code directory not found")

    print("[SETTING UP FOR CAPTURE PROCESS]")

    # Set max captures to take
    NUM_CAPTURES = 1

    # Initialize the camera and set camera properties
    camera = cv.VideoCapture(-1)
    camera_frame = (CAMERA_WIDTH, CAMERA_HEIGHT) # x, y dimensions (same as projector)
    camera.set(cv.CAP_PROP_FRAME_WIDTH, camera_frame[0])
    camera.set(cv.CAP_PROP_FRAME_HEIGHT, camera_frame[1]) 
    camera.set(cv.CAP_PROP_EXPOSURE, 0)
    camera.set(cv.CAP_PROP_BUFFERSIZE, 1)

    # Setup a window to be used
    cv.namedWindow("main", cv.WINDOW_NORMAL)
    cv.setWindowProperty("main", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

    # Find all needed files
    GRAYCODE_PATHS = sorted(glob.glob(GRAYCODE_DIRECTORY + "/pattern_*.png"))

    # Construct target directories
    target_dirs = "./ScanCaptures"
    if os.path.exists(target_dirs):
        shutil.rmtree(target_dirs)
        os.mkdir(target_dirs)
    else:
        os.mkdir(target_dirs)

    print(f"WILL SAVE IMAGES TO {target_dirs}")

    # key: file name, value: image file data
    projection_patterns = {f_name: cv.imread(f_name) for f_name in GRAYCODE_PATHS}

    print("[STARTING CAPTURE PROCESS]")

    # Repeat process for each capture
    [ScanProjectAndCapture(camera,projection_patterns[file_name], j, target_dirs) for j, file_name in enumerate(projection_patterns)]

    # Release the camera
    camera.release()

### POINT CLOUD DISPLAY AND SAVE ###

def PointCloudDisplay():
    # Load the points from the file
    saved_cali = cv.FileStorage("3Dpoints.xml", cv.FILE_STORAGE_READ)
    points = saved_cali.getNode("a3D_Points").mat()
    colors = saved_cali.getNode("Colors").mat()

    #Outputs points shape and dtype for debugging
    print("Original points shape:", points.shape)
    print("Original points dtype:", points.dtype)

    # Reshape points to work with Open3D
    points_reshaped = points.reshape((-1, 3))

    # Remove infinite points
    valid_points = points_reshaped[np.isfinite(points_reshaped).all(axis=1)]

    # Convert the valid points to an Open3D PointCloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(valid_points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    # Visualize the point cloud
    o3d.io.write_point_cloud("output_point_cloud.ply", point_cloud)
    o3d.visualization.draw_geometries([point_cloud])


### MESH DISPLY AND SAVE ###

class MeshGenerator:
    """This generates a Mesh using a point cloud, letting one display and save it.
    """
    
    def __init__(self, point_cloud: np.ndarray, colors: np.ndarray, normal_estimation_point_ref: int = 100, filter_point_cloud: bool = True,
                 nb_neighbors:int = 1000, std_ratios= 1.5, rgb_colors: float = [1, 0, 0]) -> None:
        """Creates a point cloud, filters it, and then generates a Mesh

        Args:
            point_cloud (np.ndarray): The point Cloud in the form of N X 3
            colors (np.ndarray): An array of colors, normalized for values 0 to 1 in the form of N X 3
            normal_estimation_point_ref (int, optional): Ref points for normal estimation. Defaults to 100.
            filter_point_cloud (bool, optional): Decides if the point cloud should be filtered. Defaults to True.
            nb_neighbors (int, optional): Ref points for statistical filtering. Defaults to 1000.
            std_ratios (float, optional): Defines how many std devs above the mean qualifies as an outlier. Defaults to 1.5.
            rgb_colors (float, optional): Color to pain the mesh. Defaults to [1, 0, 0].
        """
        # Setup the point cloud in an open3d friendly format, then insert points and colors
        self.point_cloud = o3d.geometry.PointCloud()
        self.point_cloud.points = o3d.utility.Vector3dVector(point_cloud)
        self.point_cloud.colors =  o3d.utility.Vector3dVector(colors)
        
        if filter_point_cloud:
            # down samples, then removes outlier
            self.point_cloud, ind = self.point_cloud.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratios)
        
        self.point_cloud.estimate_normals()
        # The number is the number of points to reference when creating normals
        self.point_cloud.orient_normals_consistent_tangent_plane(normal_estimation_point_ref)
        
        # Generate the mesh
        self.mesh, self.densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(self.point_cloud)
        
        # Paint and compute vertex normals for a better view
        self.mesh.compute_vertex_normals()
        self.mesh.paint_uniform_color(rgb_colors)
        
        
    

    def SaveMesh(self, file_path_mesh: str, file_path_point_cloud: str = None) -> None:
        """Saves the Mesh and the point cloud (optionally)

        Args:
            file_path_mesh (str): File to write mesh to
            file_path_point_cloud (str): File to write point cloud to. Defaults to None - nothing is written.
        """

        isWritten = o3d.io.write_triangle_mesh(file_path_mesh, self.mesh)
        
        if not isWritten:
            print(f"The file {file_path_mesh} failed to write.")
            
        if file_path_point_cloud:
            isWritten = o3d.io.write_point_cloud(file_path_point_cloud, self.point_cloud)
            
            if not isWritten:
                print(f"The file {file_path_mesh} failed to write.")

    def DisplayMesh(self, showPointCloudToo: bool = False) -> None:
        """Displays the Mesh and optionally the point cloud.

        Args:
            showPointCloudToo (bool, optional): Decide if the mesh should be shown. Defaults to False.
        """
        if showPointCloudToo:
            o3d.visualization.draw_geometries([self.point_cloud, self.mesh])
        else :
            o3d.visualization.draw_geometries([self.mesh])


def GenMesh():

    file_path = "mesh.ply"
    pcd_file_path = "filtered_pcd.ply"
    saved_cali = cv.FileStorage("3Dpoints.xml", cv.FILE_STORAGE_READ)
    colors = saved_cali.getNode("Colors").mat()
    points = saved_cali.getNode("a3D_Points").mat()
    mesh = MeshGenerator(points, colors)
    mesh.SaveMesh(file_path, pcd_file_path)
    mesh.DisplayMesh(True)
