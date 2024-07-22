import os
from PIL import Image
import pytesseract

# Set the TESSDATA_PREFIX environment variable
os.environ['TESSDATA_PREFIX'] = '/opt/anaconda3/envs/ocv/share/tessdata'

# Specify the Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = '/opt/anaconda3/envs/ocv/bin/tesseract'

print(pytesseract.image_to_string('../Data/test.png'))
print(pytesseract.image_to_string('../Data/test-european.jpg'))



