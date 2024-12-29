import SLC_Library
import cv2 as cv
import glob
import os
import time
import shutil

# 
# Change all of these variables to fit your System
#

proj_height = 720 #Height in pixels of the projector
proj_width = 1280 #Width in pixels of the projector
cam_height = 1080 #Height in pixels of the camera
cam_width = 1920 #Width in pixels of the camera
cvv = 6 #Number of vertices of the checkerboard on the verticle plane
chv = 8 #Number of vertices of the checkerboard on the horizontal plane
csize = 25 #Size of checkerboard squares in milimeters
graycode_step = 1 #Step size of the graycode, this should not have to change unless you are testing different orientations of patterns.
os.system("v4l2-ctl -d /dev/video* --set-ctrl=exposure_auto=1") #this sets the webcams auto exposure to off
if os.path.exists("./graycode_pattern"):
    print("Graycode pattern folder already exists, assuming user is just recalibrating with the same system. If this is not the case please delete the graycode_pattern folder and try again")
    time.sleep(5)
    PatternsCreated = False
else:
    print("No graycode images detected, generating new ones based on variables")
    SLC_Library.GenGraycodeImgs_main(proj_height,proj_width,graycode_step)
    PatternsCreated = False

print("Generation Complete, now taking images of checkerboard with projections")
SLC_Library.cap_sequence_main()
#Take the images of the projection on the checkerboard pattern, 
#and saves them into capture_0 to capture_(number of captures) default is 5 but can be changed
print("Images complete, now calibrating based on images")
SLC_Library.calibrate_main(proj_height,proj_width,cvv,chv,csize,graycode_step)
#Takes the variables and outputs a calibration xml with all of the coefficients that will be used
#to undistort images used in the scanning, and also to create the depth map and point cloud
print("Calibration complete")
#Takes the graycode patterns used in the original calibration and undistorts them according to the intrisics
#obtained from the calibration and outputs these images to be used in the ScanMain