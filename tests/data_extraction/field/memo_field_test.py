import unittest

from backend.data_extraction.field.memo_field import MemoField
from backend.data_extraction.field.data import field_data

"""
This class tests the MemoField class
It extends the unittest TestCase class
"""

class MyTestCase(unittest.TestCase):
    """
    Tests if string of letter is sent, there should be no problems parsing it

    @param self: self sent to method

    Pass if memo_field.validate returns True
    """

    def test_is_alpha(self):
        print("\tMemo Validation Test 1")
        test_pay_to_order_field_class = MemoField
        test_pair = field_data.FieldData

        test_pair.field_type = field_data.FieldType.FIELD_TYPE_MEMO
        test_pair.extracted_data = "Tomas"
        self.assertEqual(test_pay_to_order_field_class.validate(MemoField, test_pair), True)

    """
    Tests if string of numbers is sent, there should be no problems parsing it

    @param self: self sent to method

    Pass if memo_field.validate returns True
    """

    def test_is_letter(self):
        print("\tMemo Validation Test 2")
        test_pay_to_order_field_class = MemoField
        test_pair = field_data.FieldData

        test_pair.field_type = field_data.FieldType.FIELD_TYPE_MEMO
        test_pair.extracted_data = "124341"
        self.assertEqual(test_pay_to_order_field_class.validate(MemoField, test_pair), True)

    """
    Tests if digit number is sent, there should be no problems parsing it

    @param self: self sent to method

    Pass if memo_field.validate returns False
    """

    def test_is_not_string(self):
        print("\tMemo Validation Test 3")
        test_pay_to_order_field_class = MemoField
        test_pair = field_data.FieldData

        test_pair.field_type = field_data.FieldType.FIELD_TYPE_MEMO
        test_pair.extracted_data = 12345
        self.assertEqual(test_pay_to_order_field_class.validate(MemoField, test_pair), False)

if __name__ == '__main__':
    unittest.main()