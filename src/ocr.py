import os
from PIL import Image
import pytesseract

# Set the TESSDATA_PREFIX environment variable
os.environ['TESSDATA_PREFIX'] = '/opt/anaconda3/envs/ocv/share/tessdata'

# Specify the Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = '/opt/anaconda3/envs/ocv/bin/tesseract'


# print(pytesseract.image_to_string('../Data/test.png'))
print(pytesseract.image_to_string('../data/chosen-images/1000-receipt.jpg'))


def get_text(image):
    """
    Use pytesseract to extract text from the image

    Parameter
        :param image: processed image
        :return:
    Config options for tesseract:
        https://tesseract-ocr.github.io/tessdoc/Command-Line-Usage.html#simplest-invocation-to-ocr-an-image
    - custom_config = r'--oem 3 --psm 3'
    pytesseract.image_to_string(image, custom_config)
    - Didnt rly work, does its job in Default
    """

    text = pytesseract.image_to_string(image)
    return text
