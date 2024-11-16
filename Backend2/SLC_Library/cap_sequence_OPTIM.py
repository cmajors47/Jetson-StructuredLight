import cv2 as cv
import os
import shutil
import glob
import subprocess
import time

def projectAndCapture(camera,proj_img, pattern_count, target_dir):
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
    
    cv.imwrite(f"{target_dir}/graycode_{pattern_count:02d}.png", img)
    return

def main():
    # Set max captures to take
    num_captures = 3

    # Initialize the camera and set camera properties
    camera = cv.VideoCapture(-1)
    camera_frame = (1920, 1080) #x,y dimensions (same as projector)
    camera.set(cv.CAP_PROP_FRAME_WIDTH, camera_frame[0])
    camera.set(cv.CAP_PROP_FRAME_HEIGHT, camera_frame[1]) 
    camera.set(cv.CAP_PROP_EXPOSURE, -2)
    # Note that the internal buffer will not be updated. Two reads need to be done to get one photo - the first to clear the buffer, the second
    # to get the most recent input. Check Link: https://www.reddit.com/r/opencv/comments/p415cc/question_how_do_i_get_a_fresh_frame_taken_after/
    camera.set(cv.CAP_PROP_BUFFERSIZE, 1)

    # Setup a window to be used
    cv.namedWindow("main", cv.WINDOW_NORMAL)
    cv.setWindowProperty("main", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

    # Find all needed files
    graycode_names = sorted(glob.glob("graycode_pattern/pattern_*.png"))
    capture_folders = glob.glob("capture_*/")
    # Remove directories with same names
    for folder_name in capture_folders:
        shutil.rmtree(folder_name)
    # Construct target directories
    target_dirs = [f"capture_{c}" for c in range(0, num_captures)]
    for file in target_dirs:
        os.mkdir(file)

    # key: file name, value: image file data
    projection_patterns = {f_name: cv.imread(f_name) for f_name in graycode_names}

    # Repeat process for each capture
    for i, dir in enumerate(target_dirs):
        [projectAndCapture(camera,projection_patterns[file_name], j, dir) for j, file_name in enumerate(projection_patterns)]
        print(f"There are {num_captures - i} remaining capture(s). Adjust target and press any key to continue.")
        cv.waitKey(0)

    # Release the camera
    camera.release()
