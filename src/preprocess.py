import cv2
import numpy as np

def analyze_histogram(image):
    """
    Analyzes the image histogram to decide if brightness or contrast adjustments are needed
    Args: image (numpy.ndarray): Input grayscale image.
    If needed: returns True, else returns False.
    """
    #Calc histogram
    hist = cv2.calcHist([image], [0], None, [256], [0,256])
    hist = hist / hist.sum()
    cdf = hist.cumsum()

    low_threshold = 0.01
    high_threshold = 0.99

    low_percentile = np.argmax(cdf > low_threshold)
    high_percentile = np.argmax(cdf > high_threshold)

    if low_percentile > 10:
        return True
    return False

def adjust_brightness_contrast(image, alpha=1.5, beta=-40):
    if analyze_histogram(image):
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return image

def binaryization(img):
    result_img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return result_img

def preprocess_image(img):
    # Check if the image is already in grayscale
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Generate Image Histogram
    image = adjust_brightness_contrast(img)
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
    
    cv2.imshow("original", img)
    cv2.waitKey(0)
    cv2.imshow("adjusted", image)
    cv2.waitKey(0)

    # Return Image
    # return binaried_image

image_path = f'../data/images/1002-receipt.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
preprocess_image(image)
