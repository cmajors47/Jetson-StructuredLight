import SLC_Library.scanargs_OPTIM
import SLC_Library.Distort_Tools
import glob
import cv2 as cv
import shutil
import os

proj_height = 720 #Height in pixels of the projector
proj_width = 1280 #Width in pixels of the projector
cam_height = 1080 #Height in pixels of the camera
cam_width = 1920 #Width in pixels of the camera
cvv = 6 #Number of vertices of the checkerboard on the verticle plane
chv = 8 #Number of vertices of the checkerboard on the horizontal plane
csize = 25 #Size of checkerboard squares in milimeters
graycode_step = 1 #Step size of the graycode, this should not have to change unless you are testing different orientations of patterns.
dir = "./captureUndistortedOptim"
SLC_Library.scanargs_OPTIM.main(cam_width,cam_height)

object = SLC_Library.Distort_Tools.CalibrationResults("calibration_result.xml")
graycode_names = sorted(glob.glob("captureOptimized_*/capture_*"))
pattern_count = 0
if os.path.exists(dir):
    shutil.rmtree(dir)
    os.mkdir(dir)
else:
    os.mkdir(dir)
for file_name in graycode_names:
    # Load the image from the file path
    img = cv.imread(file_name, cv.IMREAD_GRAYSCALE)  # Use cv.IMREAD_GRAYSCALE if needed
    if img is None:
        print(f"Error loading image: {file_name}")
        continue
    
    # Undistort the image using the calibration results
    img_undistorted = object.undistort_camera_img(img)
    
    # Save the undistorted image
    cv.imwrite(f"captureUndistortedOptim/capture_{pattern_count:02d}.png", img_undistorted)
    
    pattern_count += 1