import crop
import glob
corners = crop.find_white_corners("/home/slc/SLC/Backend/procam-calibration/capturesUndistorted/capture_42.png")
crop.crop_image("/home/slc/SLC/Backend/procam-calibration/capturesUndistorted/capture_*.png", corners)
