import pytesseract
import cv2
from PIL import Image
import numpy as np


def extract_data_py_tesseract(image):
    pytesseract.pytesseract.tesseract_cmd = r'C:\Users\development\AppData\Local\Tesseract-OCR\tesseract.exe'

    filename = "{}.png".format("temp")
    cv2.imwrite(filename, image)
    text = pytesseract.image_to_string(Image.open(filename), config='--psm 7')

    return text

