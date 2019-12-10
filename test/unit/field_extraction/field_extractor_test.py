import unittest
import os
import cv2
from src.main.backend.preprocess.preprocess_main import preprocessEntryPoint
from src.main.backend.field_extraction.field_extractor_main import extractFieldsEntryPoint

class FieldExtractorTest(unittest.TestCase):

    def test_all_fields_extracted(self):
        filedir = os.path.abspath(os.path.dirname(__file__))
        img_path = os.path.join(filedir, '../../test-files/unit_images/test_good_text_good_bg2.png')
        img = cv2.imread(img_path)
        prp, old = preprocessEntryPoint(img)
        nimg, fields = extractFieldsEntryPoint(old, prp)

        self.assertEqual(8, len(fields))


if __name__ == '__main__':
    unittest.main()
