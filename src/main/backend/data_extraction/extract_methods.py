from PIL import Image
import sys
import pyocr.builders
import cv2
import pytesseract
from .handwriting_extract.src.main import extract

"""
Extracts data from the provided openCV image using the pyocr Tesseract wrapper.

@param image: openCV image

@return Dictionary containing the extracted text and mean confidence of the text
"""

def extract_data_pyocr(image):
    tools = pyocr.get_available_tools()
    if len(tools) == 0:
        print("No OCR tool found")
        sys.exit(1)

    tool = tools[0]

    langs = tool.get_available_languages()
    print(langs)
    lang = langs[1]

    filename = "{}.png".format("temp")
    cv2.imwrite(filename, image)

    line_and_word_boxes = tool.image_to_string(Image.open(filename),
                                               lang=lang,
                                               builder=pyocr.builders.LineBoxBuilder())

    conf_sum = 0
    conf_count = 0
    for i, x in enumerate(line_and_word_boxes):
        for j, y in enumerate(x.word_boxes):
            conf_sum += y.confidence
            conf_count += 1

    if conf_count == 0:
        mean_conf = 0
    else:
        mean_conf = conf_sum / conf_count

    return {"text": tool.image_to_string(Image.open(filename), lang=lang, builder=pyocr.builders.TextBuilder()),
            "mean_conf": mean_conf}


def extract_data_pytesseract(image):
    filename = "{}.png".format("temp")
    cv2.imwrite(filename, image)
    text = pytesseract.image_to_string(Image.open(filename), config='--psm 7')

    return text


def extract_data_handwriting(image):
    return extract(image)


