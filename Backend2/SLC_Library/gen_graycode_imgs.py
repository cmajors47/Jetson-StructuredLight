#coding: UTF-8

import os
import os.path
import argparse
import cv2
import numpy as np

TARGET_DIR = './graycode_pattern'
CAPTURED_DIR = './capture_*' 

def constructImage(height, width, pattern, step):
    img = np.zeros((height, width), np.uint8)
    #for y in range(height):
        #for x in range(width):
        #    img[y, x] = pattern[int(y/step), int(x/step)]
    # List comprehension is faster than for loop
    img[:] = [[pattern[int(y/step), int(x/step)] for x in range(width)] for y in range(height)]
    return img

def main(proj_height,proj_width,graycode_step):
    # parser = argparse.ArgumentParser(
    #     description='Generate graycode pattern images')
    # parser.add_argument('proj_height', type=int, help='projector pixel height')
    # parser.add_argument('proj_width', type=int, help='projector pixel width')
    # parser.add_argument('-graycode_step', 
    #                     type=int, default=1,
    #                     help='step size of Gray Code [default:1](increase if moire appears)')

    step = graycode_step
    height = proj_height
    width = proj_width
    gc_height = int((height - 1 ) / step) + 1
    gc_width = int((width - 1) / step) + 1
    
    graycode = cv2.structured_light.GrayCodePattern.create(gc_width, gc_height)
    patterns = graycode.generate()[1]
    
    # Expand image size
    exp_patterns = [constructImage(height, width, pat, step) for pat in patterns]
    # Append white and black patterns
    exp_patterns.append(np.full((height, width), fill_value=255, dtype=np.uint8))
    exp_patterns.append(np.zeros((height, width), dtype=np.uint8))

    if not os.path.exists(TARGET_DIR):
        os.mkdir(TARGET_DIR)

    for i, pat in enumerate(exp_patterns):
        #cv2.imwrite(TARGET_DIR + '/pattern_' + str(i).zfill(2) + '.png', pat)
        cv2.imwrite(f"{TARGET_DIR}/pattern_{str(i).zfill(2)}.png", pat)

    output = (
        f"=== Result ===\n"
        f"\'{TARGET_DIR}/pattern_00.png ~ pattern_{str(len(exp_patterns)-1)}.png\' were generated.\n\n"
        f"=== Next step ===\n"
        f"Project patterns and save captured images as \'{CAPTURED_DIR}/graycode_*.png\'\n\n"
        f"    ./ --- capture_1/ --- graycode_00.png\n"
        f"        |              |- graycode_01.png\n"
        f"        |              |        .\n"
        f"        |              |        .\n"
        f"        |              |- graycode_{str(len(exp_patterns)-1)}.png\n"
        f"        |- capture_2/ --- graycode_00.png\n"
        f"        |              |- graycode_01.png\n"
        f"        |      .       |        .\n"
        f"        |      .       |        .\n\n"
        f"It is recommended to capture more than 5 times.\n"
    )
    print(output)

if __name__ == '__main__':
    main()
