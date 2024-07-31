import cv2
import numpy as np
import matplotlib.pyplot as plt


def analyze_histogram(image):
    """
    Analyzes the image histogram to decide if brightness or contrast adjustments are needed
    Args: image (numpy.ndarray): Input grayscale image.
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
    result_img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
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
        noise_removed = cv2.medianBlur(image, 3)
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

    # Binarization of best equalized image
    binaried_image = binaryization(equalized_image)
    # canny_img = canny_edge_detection(binaried_image)
    cv2.imshow("bi img", binaried_image)
    cv2.waitKey(0)
    return binaried_image

# image_path = f'../data/images/1002-receipt.jpg'
# image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
# preprocess_image(image_gray)


def canny_edge_detection(image):
    """
    Perform Canny edge detection on a denoised, binarized image.

    Args:
        image (numpy.ndarray): The denoised, binarized image.

    Returns:
        numpy.ndarray: The image with edges detected.
    """
    # Define the lower and upper thresholds for the Canny edge detector
    lower_threshold = 100
    upper_threshold = 150

    # Apply the Canny edge detector
    edges = cv2.Canny(image, lower_threshold, upper_threshold)
    cv2.imshow("bi img", edges)
    cv2.waitKey(0)
    return edges
