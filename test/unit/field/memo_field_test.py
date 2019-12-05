import unittest

from src.main.backend.data_extraction.field.memo_field import MemoField
from src.main.backend.data_extraction.field.data.field_data import FieldData, FieldType

"""
This class test the MemoField class
It extends the unittest TestCase class
"""

class MemoTest(unittest.TestCase):
    """
    Tests if string of letter is sent, there should be no problems parsing it

    @param self: self sent to method

    Pass if memo_field.validate returns True
    """

    def test_is_alpha(self):
        print("\tMemo Validation Test 1")
        test_pay_to_order_field_class = MemoField()
        test_pair = FieldData()

        test_pair.field_type = FieldType.FIELD_TYPE_MEMO
        test_pair.extracted_data = "Tomas"
        self.assertEqual(test_pay_to_order_field_class.validate(test_pair), True)

    """
    Tests if string of numbers is sent, there should be no problems parsing it

    @param self: self sent to method

    Pass if memo_field.validate returns True
    """

    def test_is_letter(self):
        print("\tMemo Validation Test 2")
        test_pay_to_order_field_class = MemoField()
        test_pair = FieldData()

        test_pair.field_type = FieldType.FIELD_TYPE_MEMO
        test_pair.extracted_data = "124341"
        self.assertEqual(test_pay_to_order_field_class.validate(test_pair), True)

    """
    Tests if digit number is sent, there should be no problems parsing it

    @param self: self sent to method

    Pass if memo_field.validate returns True
    """

    def test_is_not_string(self):
        print("\tMemo Validation Test 3")
        test_pay_to_order_field_class = MemoField()
        test_pair = FieldData()

        test_pair.field_type = FieldType.FIELD_TYPE_MEMO
        test_pair.extracted_data = 12345
        self.assertEqual(test_pay_to_order_field_class.validate(test_pair), True)

    """
        Tests if empty string is sent, it should be rejected

        @param self: self sent to method

        Pass if memo_field.validate returns False
        """

    def test_is_empty(self):
        print("\tMemo Validation Test 4")
        test_pay_to_order_field_class = MemoField()
        test_pair = FieldData()

        test_pair.field_type = FieldType.FIELD_TYPE_MEMO
        test_pair.extracted_data = ""
        self.assertEqual(test_pay_to_order_field_class.validate(test_pair), False)


if __name__ == '__main__':
    unittest.main()