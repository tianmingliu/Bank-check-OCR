from PIL import Image
import sys
import pyocr.builders
import cv2
import pytesseract
import backend.data_extraction.handwriting_extract.src.main as hw
from backend.data_extraction.handwriting_extract.src.DataLoader import FilePaths
from backend.data_extraction.handwriting_extract.src.Model import DecoderType
from backend.data_extraction.handwriting_extract.src.Model import Model
import tensorflow as tf

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
    e1 = extract_data_pyocr(image)
    e2 = extract_data_pytesseract(image)
    e3 = hw.extract(image)

    print("PyOCR: " + e1['text'])
    print("PyTesseract: " + e2)
    print("Handwriting: " + e3)

    return e2


if __name__ == "__main__":
    # img = cv2.imread("resources/images/amount_test.png")
    # img = cv2.imread("resources/images/company_test.png")
    # img = cv2.imread("resources/images/company_test_ul.png")
    # img = cv2.imread("resources/images/date_test.png")    # ALL FAIL
    # img = cv2.imread("resources/images/date_test2.png")   # ALL FAIL
    # img = cv2.imread("resources/images/hello_test.png")
    # img = cv2.imread("resources/images/money_test.png")
    img = cv2.imread("resources/images/money_test_ul.png")  # ALL FAIL
    # img = cv2.imread("resources/images/word_test_ul.png")

    e1 = extract_data_pyocr(img)
    e2 = extract_data_pytesseract(img)
    e3 = extract_data_handwriting(img)

    print("PyOCR: " + e1['text'])
    print("PyTesseract: " + e2)
    print("Handwriting: " + e3)

