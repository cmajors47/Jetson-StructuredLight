import SLC_Library
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
dir = "./ScanCaptures"
SLC_Library.Scan_main(cam_width,cam_height)
SLC_Library.GenPointCloud()
SLC_Library.PointCloudDisplay()
SLC_Library.GenMesh()
