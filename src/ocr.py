import os
from PIL import Image
import pytesseract

# Set the TESSDATA_PREFIX environment variable
os.environ['TESSDATA_PREFIX'] = '/opt/anaconda3/envs/ocv/share/tessdata'

# Specify the Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = '/opt/anaconda3/envs/ocv/bin/tesseract'


# print(pytesseract.image_to_string('../Data/test.png'))
# print(pytesseract.image_to_string('../Data/test-european.jpg'))

def get_text(image):
    # Use pytesseract to extract text from the image
    text = pytesseract.image_to_string(image)
    return text
