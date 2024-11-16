import cv2
import numpy as np
import glob
import Distort_Tools

def find_white_corners(image_path):
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Ensure the image was loaded correctly
    if image is None:
        print("Error loading image.")
        return None
    
    # Threshold the image to get a binary image (white as 255, all other pixels as 0)
    _, binary_image = cv2.threshold(image, 25, 255, cv2.THRESH_BINARY)

    # Get the image dimensions
    height, width = binary_image.shape
    
    # Initialize coordinates for the four farthest white corners
    top_left = None
    top_right = None
    bottom_left = None
    bottom_right = None

    # Start searching from the outermost pixels towards the center
    for y in range(height):
        for x in range(width):
            if binary_image[y, x] ==255:  # White pixel found
                # Update top-left corner
                if top_left is None or (x < top_left[0] and y < top_left[1]):
                    top_left = (x, y)
                # Update top-right corner
                if top_right is None or (x > top_right[0] and y < top_right[1]):
                    top_right = (x, y)
                # Update bottom-left corner
                if bottom_left is None or (x < bottom_left[0] and y > bottom_left[1]):
                    bottom_left = (x, y)
                # Update bottom-right corner
                if bottom_right is None or (x > bottom_right[0] and y > bottom_right[1]):
                    bottom_right = (x, y)

    if top_left and top_right and bottom_left and bottom_right:
        print(f"Top Left: {top_left}, Top Right: {top_right}, Bottom Left: {bottom_left}, Bottom Right: {bottom_right}")
        return top_left, top_right, bottom_left, bottom_right
    else:
        print("No white corners found!")
        return None

def crop_image(image_path, corners, output_prefix="CapturesCropped/capture_"):
    # Load the original image
    img_number = 0
    top_left, top_right, bottom_left, bottom_right = corners
    file_names = sorted(glob.glob(image_path))
    min_x = max_x = min_y = max_y = 0
    image = None
    for img in file_names:
        image = cv2.imread(img)
        # Define the bounding box for cropping
        min_x = min(top_left[0], bottom_left[0])
        max_x = max(top_right[0], bottom_right[0])
        min_y = min(top_left[1], top_right[1])
        max_y = max(bottom_left[1], bottom_right[1])

        # Crop the image based on the found corners
        cropped_image = image[min_y:max_y, min_x:max_x]

        # Save the cropped image
        output_file = f"{output_prefix}{img_number}.png"
        cv2.imwrite(output_file, cropped_image)
        print(f"Cropped image saved as {output_file}")
        img_number += 1
    
    with open("/home/slc/SLC/Backend/procam-calibration/Utilities/new_img_dimensions.txt", "w") as file:
        file.write(f"{int(max_x - min_x)}, {int(max_y - min_y)}")
    
    # Load intrinsic parameters
    calib_data = Distort_Tools.CalibrationResults("/home/slc/SLC/Backend/procam-calibration/calibration_result.xml")
    K_cam = calib_data.cam_int
    K_proj = calib_data.proj_int

    # Shift Principal Point Cx, Cy
    cx_new = K_cam[0, 2] - min_x
    cy_new = K_cam[1, 2] - min_y

    # Scale Focal Length
    s = (max_x - min_x) / image.shape[1]
    fx_new = s * K_cam[0, 0]
    fy_new = s * K_cam[1, 1]

    K_cam_new = np.array([
        [fx_new, K_cam[0, 1], cx_new],
        [0, fy_new, cy_new],
        [0, 0, 1]
    ])

    return K_cam_new

def main(image_path):
    # Find the white corners of the image
    corners = find_white_corners(image_path)

    if corners:
        # Crop the image based on the found corners
        crop_image(image_path, corners)
    
if __name__ == "__main__":
    # Example usage, provide your image path here
    image_path = "your_image.png"
    main(image_path)