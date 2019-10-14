import unittest
import os
import cv2
import backend.preprocess.preprocess_main as prp
import backend.data_extraction.digit_recognition.pyocr_ocr.handwriting_extract as extract


class HandwritingExtractTest(unittest.TestCase):
    @staticmethod
    def get_preprocessed_file(file_name : str):
        filedir = os.path.abspath(os.path.dirname(__file__))
        filename = os.path.join(filedir, '..\\test_images\\' + file_name)
        image = cv2.imread(filename)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        return gray

    def test_digital_text(self):
        data = extract.extract_data(self.get_preprocessed_file("cropped_field9.jpg"))
        print(data)
        self.assertNotEqual("", data['text'], "Text that was read in was blank.")
        self.assertNotEqual(0, data['mean_conf'], "Mean confidence was not recorded")
        self.assertEqual("Lady Gaga", data['text'], "Text was read in incorrectly")

    def test_handwriting_text(self):
        data = extract.extract_data(self.get_preprocessed_file("test3.png"))
        print(data)
        self.assertNotEqual("", data['text'], "Text that was read in was blank.")
        self.assertNotEqual(0, data['mean_conf'], "Mean confidence was not recorded")
        self.assertEqual("Hello.", data['text'], "Text was read in incorrectly")

    def test_digits(self):
        data = extract.extract_data(self.get_preprocessed_file("cropped_field7.jpg"))
        print(data)
        self.assertNotEqual("", data['text'], "Text that was read in was blank.")
        self.assertNotEqual(0, data['mean_conf'], "Mean confidence was not recorded")
        self.assertEqual("975.50", data['text'], "Text was read in incorrectly")

    def test_digits_and_letters(self):
        data = extract.extract_data(self.get_preprocessed_file("cropped_field6.jpg"))
        print(data)
        self.assertNotEqual("", data['text'], "Text that was read in was blank.")
        self.assertNotEqual(0, data['mean_conf'], "Mean confidence was not recorded")


if __name__ == '__main__':
    unittest.main()
