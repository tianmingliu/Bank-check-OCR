from PIL import Image
import cv2
import pytesseract
from .handwriting_extract.src.main import extract


def extract_data_pytesseract(image):
    filename = "{}.png".format("temp")
    cv2.imwrite(filename, image)
    text = pytesseract.image_to_string(Image.open(filename), config='--psm 7')

    return text


def extract_data_handwriting(image):
    return extract(image)


