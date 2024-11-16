import SLC_Library.Distort_Tools
import SLC_Library
import SLC_Library.cap_sequence_OPTIM
import SLC_Library.gen_graycode_imgs
import cv2 as cv
import glob
import os
import time
import SLC_Library.calibrate
import shutil

proj_height = 720 #Height in pixels of the projector
proj_width = 1280 #Width in pixels of the projector
cam_height = 1080 #Height in pixels of the camera
cam_width = 1920 #Width in pixels of the camera
cvv = 6 #Number of vertices of the checkerboard on the verticle plane
chv = 8 #Number of vertices of the checkerboard on the horizontal plane
csize = 25 #Size of checkerboard squares in milimeters
graycode_step = 1 #Step size of the graycode, this should not have to change unless you are testing different orientations of patterns.

if os.path.exists("./graycode_pattern"):
    print("Graycode pattern folder already exists, assuming user is just recalibrating with the same system. If this is not the case please delete the graycode_pattern folder and try again")
    time.sleep(5)
    PatternsCreated = False
else:
    print("No graycode images detected, generating new ones based on variables")
    SLC_Library.gen_graycode_imgs.main(proj_height,proj_width,graycode_step)
    PatternsCreated = False

print("Generation Complete, now taking images of checkerboard with projections")
SLC_Library.cap_sequence_OPTIM.main()
#Take the images of the projection on the checkerboard pattern, 
#and saves them into capture_0 to capture_(number of captures) default is 3 but can be changed
print("Images complete, now calibrating based on images")
SLC_Library.calibrate.main(proj_height,proj_width,cvv,chv,csize,graycode_step)
#Takes the variables and outputs a calibration xml with all of the coefficients that will be used
#to undistort images used in the scanning, and also to create the depth map and point cloud
print("Calibration complete, now undistorting original patterns according to calibration")
object = SLC_Library.Distort_Tools.CalibrationResults("calibration_result.xml")
graycode_names = sorted(glob.glob("graycode_pattern/pattern_*"))
pattern_count = 0
for file_name in graycode_names:
    # Load the image from the file path
    img = cv.imread(file_name, cv.IMREAD_GRAYSCALE)  # Use cv.IMREAD_GRAYSCALE if needed
    if img is None:
        print(f"Error loading image: {file_name}")
        continue
    
    # Undistort the image using the calibration results
    img_undistorted = object.undistort_proj_img(img)
    
    # Save the undistorted image
    cv.imwrite(f"graycode_undistorted/pattern_{pattern_count:02d}.png", img_undistorted)
    
    pattern_count += 1
#Takes the graycode patterns used in the original calibration and undistorts them according to the intrisics
#obtained from the calibration and outputs these images to be used in the ScanMain