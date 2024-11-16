import cv2 as cv
import os
import shutil
import glob
import subprocess
import time
import sys

def projectAndCapture(camera, proj_img, pattern_count, target_dir):
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

def main(CAMERA_WIDTH, CAMERA_HEIGHT):

    os.system("export DISPLAY=:0")

    # Get command line arguments
    GRAYCODE_DIRECTORY = "./graycode_undistorted"

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
    target_dirs = [f"./captureOptimized_{c}" for c in range(0, NUM_CAPTURES)]
    for dir in target_dirs:
        if os.path.exists(dir):
            shutil.rmtree(dir)
            os.mkdir(dir)
        else:
            os.mkdir(dir)

    print(f"WILL SAVE IMAGES TO {target_dirs}")

    # key: file name, value: image file data
    projection_patterns = {f_name: cv.imread(f_name) for f_name in GRAYCODE_PATHS}

    print("[STARTING CAPTURE PROCESS]")

    # Repeat process for each capture
    for i, dir in enumerate(target_dirs):
        [projectAndCapture(camera,projection_patterns[file_name], j, dir) for j, file_name in enumerate(projection_patterns)]
        print(f"Capture ({i+1} of {NUM_CAPTURES}) complete")

    # Release the camera
    camera.release()
