import Distort_Tools
import glob
import cv2 as cv

object = Distort_Tools.CalibrationResults("/home/slc/SLC/Backend/procam-calibration/calibration_result.xml")
graycode_names = sorted(glob.glob("/home/slc/SLC/Backend/procam-calibration/graycode_pattern/pattern_*.png"))
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
    cv.imwrite(f"graycode_undistorted/graycode_{pattern_count}.png", img_undistorted)
    
    pattern_count += 1