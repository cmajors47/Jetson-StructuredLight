import cv2
import numpy as np
import sys
#import calibration

calib_parameters_path = "/home/slc/SLCOptim/calibration_result.xml"
calib_params = cv2.FileStorage(calib_parameters_path, cv2.FILE_STORAGE_READ)

img_shape = calib_params.getNode("img_shape").mat().flatten()
cam_int = calib_params.getNode("cam_int").mat()
cam_dist = calib_params.getNode("cam_dist").mat()
proj_int = calib_params.getNode("proj_int").mat()
proj_dist = calib_params.getNode("proj_dist").mat()
rotation = calib_params.getNode("rotation").mat()
translation = calib_params.getNode("translation").mat()
cam_rvecs = calib_params.getNode("cam_rvecs").mat()
cam_tvecs = calib_params.getNode("cam_tvecs").mat()
cam_objps_list = calib_params.getNode("cam_objps_list").mat()
cam_corners_list2 = calib_params.getNode("cam_corners_list2").mat()
calib_params.release()

#objpoints_n = calibration.objpoints
#imgpoints_n = calibration.imgpoints
#cam_rvecs_n = calibration.cam_rvecs
#cam_tvecs_n = calibration.cam_tvecs
#cam_int_n = calibration.cam_int
#cam_dist_n = calibration.cam_dist

print(cam_objps_list)
mean_error = 0
for i in range(len(cam_objps_list)):
    imgpoints2, _ = cv2.projectPoints(cam_objps_list[i], cam_rvecs[i], cam_tvecs[i], cam_int, cam_dist)
    error = cv2.norm(cam_corners_list2[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error
 
print( "total error: {}".format(mean_error/len(cam_objps_list)) )