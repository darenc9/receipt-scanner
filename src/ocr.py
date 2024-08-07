import os
from PIL import Image
import pytesseract

# Set the TESSDATA_PREFIX environment variable
# os.environ['TESSDATA_PREFIX'] = '/opt/anaconda3/envs/ocv/share/tessdata'

# Specify the Tesseract executable path
# For our Mac Execution
# pytesseract.pytesseract.tesseract_cmd = '/opt/anaconda3/envs/ocv/bin/tesseract'
# For our Windows Execution
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

def get_text(image):
    """
    Use pytesseract to extract text from the image

    Parameter
        :param image: processed image
        :return:
    Config options for tesseract:
        https://tesseract-ocr.github.io/tessdoc/Command-Line-Usage.html#simplest-invocation-to-ocr-an-image
    """
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(image,config=custom_config)
    return text
