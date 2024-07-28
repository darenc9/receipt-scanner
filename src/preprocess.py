import cv2
import numpy as np


def binaryization(img):
    result_img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return result_img

def preprocess_image(img):
    # Check if the image is already in grayscale
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Generate Image Histogram

    # Generate MSE - Original image

    # Noise Removal

    # Get MSE - Noise Removed image
    # Generate new Image Histogram

    # Image Equalization - w/ diff masks

    # Get MSE - Each mask of equalized image

    # Binarization of best equalized image
    binaried_image = binaryization(img)

    # Display image
    cv2.imshow("Binarized Image", binaried_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Return Image
    return binaried_image
