import cv2
import numpy as np
from PIL import Image
from src.preprocess import preprocess_image
from src.ocr import get_text

image_path = "../Data/images/1005-receipt.jpg"
image_path2 = "../Data/test-european.jpg"

# read input image
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)

# obtain processed image
processed_image = preprocess_image(image)
p2 = preprocess_image(image2)

# For Testing

# Scan unprocessed image

# Scan Read processed image
extracted_data = get_text(processed_image)
print("Processed Image Text:")
print(extracted_data)

extracted_data2 = get_text(p2)
print("Processed Image Text 2:")
print(extracted_data2)

# Show converted data in text

# Convert data into preferred extension

# Saved extracted data w/ preferred name
