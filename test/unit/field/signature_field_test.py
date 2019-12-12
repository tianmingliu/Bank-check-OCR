import unittest

from src.main.backend.data_extraction.field.signature_field import SignatureField
from src.main.backend.data_extraction.field.data.field_data import FieldData, FieldType

"""
This class test the SignatureField class
It extends the unittest TestCase class
"""

class SignatureTest(unittest.TestCase):
    """
    Tests if string of letter is sent, there should be no problems parsing it

    @param self: self sent to method

    Pass if signature_field.validate returns True
    """

    def test_is_alpha(self):
        print("\tSignature Validation Test 1")
        test_pay_to_order_field_class = SignatureField()
        test_pair = FieldData()

        test_pair.field_type = FieldType.FIELD_TYPE_SIGNATURE
        test_pair.extracted_data = "Tomas"
        self.assertEqual(test_pay_to_order_field_class.validate(test_pair), True)

    """
    Tests if string of number is sent, there should be no problems parsing it

    @param self: self sent to method

    Pass if signature_field.validate returns True
    """

    def test_is_alpha(self):
        print("\tSignature Validation Test 2")
        test_pay_to_order_field_class = SignatureField()
        test_pair = FieldData()

        test_pair.field_type = FieldType.FIELD_TYPE_SIGNATURE
        test_pair.extracted_data = "12345"
        self.assertEqual(test_pay_to_order_field_class.validate(test_pair), True)

    """
    Tests if empty string is sent, there should be no problems parsing it

    @param self: self sent to method

    Pass if signature_field.validate returns True
    """

    def test_is_empty_string(self):
        print("\tSignature Validation Test 3")
        test_pay_to_order_field_class = SignatureField()
        test_pair = FieldData()

        test_pair.field_type = FieldType.FIELD_TYPE_SIGNATURE
        test_pair.extracted_data = ""
        self.assertEqual(test_pay_to_order_field_class.validate(test_pair), False)


if __name__ == '__main__':
    unittest.main()