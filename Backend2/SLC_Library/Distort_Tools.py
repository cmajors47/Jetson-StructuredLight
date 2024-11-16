import cv2 as cv
import os
import numpy as np
'''
This object provides an easy way to access the calibration results (inserted as a .xml file, with the projector and camera parameters) and
to undistort images as needed. This only requires the file path of a well structured calibration_result.xml to function.
'''

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