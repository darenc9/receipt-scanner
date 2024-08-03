import os

import cv2
import numpy as np
from PIL import Image
from fpdf import FPDF
from src.preprocess import preprocess_image
from src.ocr import get_text


def save_text_to_file(text, file_name, file_format='txt', output_folder="../Output"):
    """
    Saves the provided text to a file in the specified format at the specified location
    Parameters:
        :param text: String - text data to be saved
        :param file_name: String - Base name of the file without the extension
        :param file_format: String - Format in which to save the file. Supported formats include 'txt', 'pdf', and 'html'
        :param output_folder: String - The directory path where the file will be saved.
        :return: no return, just saves text to file
    """
    # The complete out path w/ preferred file extension
    out_path = os.path.join(output_folder, f"{file_name}.{file_format}")

    # Replace new lines with HTML break lines if the format is HTML
    text_html = text.replace("\n", "<br>")

    # Depending on preferred format, does different conversions
    if file_format.lower() == 'txt':
        with open(out_path, 'w', encoding='utf-8') as file:
            file.write(text)
    elif file_format.lower() == 'pdf':
        # Using FPDF library
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, text)
        pdf.output(out_path, 'F')
    elif file_format.lower() == 'html':
        with open(out_path, 'w', encoding='utf-8') as file:
            file.write(f'<html><body><p>{text_html}</p></body></html>')

    print(f"Text saved as {out_path}")


def process_image_file(image_path, output_format):
    """
    Processes an image file for OCR and saves the extracted text to a file
    Calling the save text to file function

    Parameters:
        :param image_path: String -  Full path to the image file
        :param output_format: String - Format to save the extracted text. Options include 'txt', 'pdf', 'html'.
        :return: Doesn't return anything, just processes and leads to saving functionality
    """
    # Reads the image into a grayscale image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Checker if none
    if image is None:
        print(f"Error: Image not found at {image_path}")
        return
    # processes the image
    pp_image = preprocess_image(image)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_filename = f"{base_name}_pp"
    save_preprocessed_img(pp_image, output_filename)
    save_text_to_file(get_text(pp_image), output_filename, output_format)


def save_preprocessed_img(img, img_name):
    """
    Saves the preprocessed image to the data/preprocessed_image/

    Parameters
        :param img: Image - Image in question
        :param img_name: String - Name of image to be saved
        :return: Doesn't return anything, just saves the image
    """
    folder_location = "../data/preprocessed_image/"
    cv2.imwrite(img, folder_location + img_name)


def main(path, output_format='txt'):
    """
    Main function to process images for OCR from a specified path.

    Parameter
        :param path: Path to an image file or a directory containing image files
        :param output_format: Desired format for the output text files ('txt', 'pdf', 'html')
    """
    # Double checks path
    if not os.path.exists(path):
        print(f"Error: The provided path does not exist: {path}")
        return
    # Checks if param is a single image or a directory of images
    if os.path.isdir(path):
        for filename in os.listdir(path):
            # Going through images of accepted extension
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                full_image_path = os.path.join(path, filename)
                # Calls process_image_file to preprocess and further the process for an image
                process_image_file(full_image_path, output_format)
    elif os.path.isfile(path):
        process_image_file(path, output_format)
    else:
        print(f"Provided path is neither a directory nor a file: {path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process images for OCR and save the extracted text.")
    # Adding arguments for the main.py
    parser.add_argument("path", help="Path to the image file or directory to process")
    parser.add_argument("--format", default='txt', choices=['txt', 'pdf', 'html'],
                        help="Format to save the extracted text")
    args = parser.parse_args()
    main(args.path, args.format)