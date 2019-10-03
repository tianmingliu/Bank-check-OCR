import pytesseract
import cv2
from PIL import Image
import numpy as np


def main():
    pytesseract.pytesseract.tesseract_cmd = r'C:\Users\development\AppData\Local\Tesseract-OCR\tesseract.exe'

    f_name = "C:\\Users\\development\\2019FallTeam03\\" \
        "backend\\data_extraction\\digit_recognition\\" \
        "test_images\\cropped_field6.jpg"

    image = cv2.imread(f_name)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    filename = "{}.png".format("temp")
    cv2.imwrite(filename, gray)
    text = pytesseract.image_to_string(Image.open(filename), config='--oem 1 --psm 7')
    print(text)


try:
    main()
except Exception as e:
    print(e.args)
    print(e.__cause__)