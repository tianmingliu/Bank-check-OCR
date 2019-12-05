import unittest
from src.main.backend.data_extraction.extract_methods import extract_data_pytesseract, extract_data_handwriting
import cv2
import os


class ExtractMethodsTest(unittest.TestCase):
    def test_pytesseract_amount(self):
        filedir = os.path.abspath(os.path.dirname(__file__))
        img1_path = os.path.join(filedir, '../../test-files/unit_images/amount_test.png')
        img1 = cv2.imread(img1_path)

        text = extract_data_pytesseract(img1)

        self.assertEqual('200.00', text)

    def test_pytesseract_date(self):
        filedir = os.path.abspath(os.path.dirname(__file__))
        img1_path = os.path.join(filedir, '../../test-files/unit_images/date_test.png')
        img1 = cv2.imread(img1_path)

        text = extract_data_pytesseract(img1)

        self.assertEqual('10/28/2019', text)

    def test_pytesseract_word(self):
        filedir = os.path.abspath(os.path.dirname(__file__))
        img1_path = os.path.join(filedir, '../../test-files/unit_images/word_test.png')
        img1 = cv2.imread(img1_path)

        text = extract_data_pytesseract(img1)

        self.assertEqual('ond.', text)

    def test_hw_amount(self):
        filedir = os.path.abspath(os.path.dirname(__file__))
        img1_path = os.path.join(filedir, '../../test-files/unit_images/amount_test.png')
        img1 = cv2.imread(img1_path)

        text = extract_data_handwriting(img1)

        self.assertEqual('I0O.0O', text)

    def test_hw_date(self):
        filedir = os.path.abspath(os.path.dirname(__file__))
        img1_path = os.path.join(filedir, '../../test-files/unit_images/date_test.png')
        img1 = cv2.imread(img1_path)

        text = extract_data_handwriting(img1)

        self.assertEqual('1O alon', text)

    def test_hw_word(self):
        filedir = os.path.abspath(os.path.dirname(__file__))
        img1_path = os.path.join(filedir, '../../test-files/unit_images/word_test.png')
        img1 = cv2.imread(img1_path)

        text = extract_data_handwriting(img1)

        self.assertEqual('Wonderland', text)


if __name__ == '__main__':
    unittest.main()
