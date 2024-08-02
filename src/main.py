import os

import cv2
import numpy as np
from PIL import Image
from src.preprocess import preprocess_image
from src.ocr import get_text

folder_path = "../data/chosen-images/"
image_path = "../data/chosen-images/1023-receipt.jpg"
# image_path2 = "../data/test-european.jpg"

# Read input image
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
# image2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)


# obtain processed image
processed_image = preprocess_image(image)
# p2 = preprocess_image(image2)

# For Testing

# Scan unprocessed image

# Scan Read processed image
extracted_data = get_text(processed_image)
print("Processed Image Text:")
print(extracted_data)

# extracted_data2 = get_text(p2)
# print("Processed Image Text 2:")
# print(extracted_data2)

# Loop through all files in the folder
# for filename in os.listdir(folder_path):
#     # Check if the file is an image
#     image_path = os.path.join(folder_path, filename)
#
#     # Read the image
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#
#     # Check if image was successfully loaded
#     if image is not None:
#         # Obtain processed image
#         processed_image = preprocess_image(image)
#
#         # Extract text from processed image
#         extracted_data = get_text(processed_image)
#
#         print(f"\n-------------------------------------")
#         print(f"Processed Image Text for {filename}:")
#         print(extracted_data)
#         print(f"\n-------------------------------------")
#     else:
#         print(f"Failed to load image: {filename}")

# Show converted data in text

# Convert data into preferred extension

# Saved extracted data w/ preferred name
