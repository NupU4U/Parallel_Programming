import os
import cv2
import numpy as np

# Set the directory paths
input_dir = "img"
output_dir = "pre-proc-img"

# Ensure the output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Function to preprocess and save image to text file
def process_image(file_path):
    # Read the image
    img = cv2.imread(file_path, 0)
    if img.shape != (28, 28):
        img = cv2.resize(img, (28, 28))

    # Reshape the image into a single flattened array
    img = img.reshape(28 * 28 * 1)

    # Normalize the image to the range 0-1
    img = img / 255.0

    # Get the file name without extension
    file_name = os.path.splitext(os.path.basename(file_path))[0]

    # Save the matrix to a text file
    output_path = os.path.join(output_dir, file_name + ".txt")
    np.savetxt(output_path, img, fmt='%1.6f')

# Process each image file in the input directory
for file_name in os.listdir(input_dir):
    if file_name.endswith(".png"):
        file_path = os.path.join(input_dir, file_name)
        process_image(file_path)
