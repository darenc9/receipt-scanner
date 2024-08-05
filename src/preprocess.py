import cv2
import numpy as np
import matplotlib.pyplot as plt


def analyze_histogram(image):
    """
    Analyzes the image histogram to decide if brightness or contrast adjustments are needed
    Parameter
        :param image: Input grayscale image.
    If needed: returns True, else returns False.
    """
    # Calc histogram
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist = hist / hist.sum()
    cdf = hist.cumsum()

    low_threshold = 0.01
    low_percentile = np.argmax(cdf > low_threshold)

    if low_percentile > 10:
        return True
    return False


def adjust_brightness_contrast(image, alpha=1.5, beta=-40):
    if analyze_histogram(image):
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return image


def binaryization(img):
    """
    Parameter
        :param img: Image to be binaried
        :return: binary image
    """
    result_img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 13)
    return result_img


def calculate_mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err


# Compares the mse value and the threshold to see if image needs to remove noise
def needs_noise_removal(image, threshold=200):
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    mse_value = calculate_mse(image, blurred)
    print(f"MSE between original and blurred image: {mse_value}")
    return mse_value > threshold


def plot_histogram(image, title):
    plt.figure()
    plt.title(title)
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    plt.hist(image.ravel(), 256, [0, 256])
    # plt.show()


# Equalize Images with CLAHE Method
def clahe_equalization(img):

    image = adjust_brightness_contrast(img)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(image)
    
    plot_histogram(image, "Original Image Histogram")
    plot_histogram(clahe_image, "CLAHE Equalized Image Histogram")
    return clahe_image


def preprocess_image(img):

    # Generate Image Histogram
    plot_histogram(img, "Original Image Histogram")

    # Enchanced brightness or contrast
    image = adjust_brightness_contrast(img)

    # Generate MSE - Original image
    mse_original = calculate_mse(image, image)
    print(f"MSE of Original Image: {mse_original}")
    # Noise Removal
    # Check if noise removal is needed
    if needs_noise_removal(image):
        # Apply noise removal
        noise_removed=cv2.bilateralFilter(image, 9, 75, 75)
    else:
        noise_removed = image
    # Get MSE - Noise Removed image
    mse_noise_removed = calculate_mse(image, noise_removed)
    print(f"MSE after Noise Removal: {mse_noise_removed}")
    # Generate new Image Histogram
    plot_histogram(noise_removed, "Histogram after Noise Removal")
    # Image Equalization - w/ CLAHE
    equalized_image = clahe_equalization(noise_removed)
    # Get MSE - Each mask of equalized image

    #Rescale Image
    scale_percent = 150  
    width = int(equalized_image.shape[1] * scale_percent / 100)
    height = int(equalized_image.shape[0] * scale_percent / 100)
    dim = (width, height)
    rescaled = cv2.resize(equalized_image, dim, interpolation=cv2.INTER_LINEAR)

    # Binarization of best equalized image
    binaried_image = binaryization(rescaled)
    #cv2.imshow("bi img", binaried_image)
    #cv2.waitKey(0)
    return binaried_image
