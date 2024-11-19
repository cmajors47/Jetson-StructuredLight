'''
The purpose of this file isi to create arrays of matrices for the purpose of collecting everything in a full scan, 
but its use may be used for other applications such as saving a group of images.
'''
import cv2 as cv
import glob
from typing import List

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
