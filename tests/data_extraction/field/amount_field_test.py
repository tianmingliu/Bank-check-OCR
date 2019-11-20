import unittest

from backend.data_extraction.field.amount_field import AmountField
from backend.data_extraction.field.data import field_data

"""
This class tests the AmountField class
It extends the unittest TestCase class
"""


class MyTestCase(unittest.TestCase):

    """
    Tests if string of random letters is sent, should fail

    @param self: self sent to method

    Pass if amount_field.validate returns False
    """
    def test_not_digit_or_written_amount(self):
        print("\tAmount Validation Test 1")
        test_amount_field_class = AmountField
        test_pair = field_data.FieldData

        test_pair.field_type = field_data.FieldType.FIELD_TYPE_AMOUNT
        test_pair.extracted_data = "aslkdj sldg hsld"
        self.assertEqual(test_amount_field_class.validate(AmountField, test_pair), False)

    """
    Tests if string of numbers is sent that are too low, should fail

    @param self: self sent to method

    Pass if amount_field.validate returns False
    """
    def test_amount_too_small(self):
        print("\tAmount Validation Test 2")
        test_amount_field_class = AmountField
        test_pair = field_data.FieldData

        test_pair.field_type = field_data.FieldType.FIELD_TYPE_AMOUNT
        test_pair.extracted_data = "0"
        self.assertEqual(test_amount_field_class.validate(AmountField, test_pair), False)

    """
    Tests if string of numbers is sent that are too high, should fail

    @param self: self sent to method

    Pass if amount_field.validate returns False
    """
    def test_amount_too_large(self):
        print("\tAmount Validation Test 3")
        test_amount_field_class = AmountField
        test_pair = field_data.FieldData

        test_pair.field_type = field_data.FieldType.FIELD_TYPE_AMOUNT
        test_pair.extracted_data = "1000000.00"
        self.assertEqual(test_amount_field_class.validate(AmountField, test_pair), False)

    """
    Tests if string of numbers is sent that are valid, should pass

    @param self: self sent to method

    Pass if amount_field.validate returns True
    """
    def test_amount_valid(self):
        print("\tAmount Validation Test 4")
        test_amount_field_class = AmountField
        test_pair = field_data.FieldData

        test_pair.field_type = field_data.FieldType.FIELD_TYPE_AMOUNT
        test_pair.extracted_data = "100.00"
        self.assertEqual(test_amount_field_class.validate(AmountField, test_pair), True)

    """
        Tests if string of written numbers is sent that are too low, should fail

        @param self: self sent to method

        Pass if amount_field.validate returns False
        """

    def test_written_amount_too_small(self):
        print("\tAmount Validation Test 5")
        test_amount_field_class = AmountField
        test_pair = field_data.FieldData

        test_pair.field_type = field_data.FieldType.FIELD_TYPE_AMOUNT
        test_pair.extracted_data = "Zero"
        self.assertEqual(test_amount_field_class.validate(AmountField, test_pair), False)

    """
    Tests if string of written numbers is sent that are too high, should fail

    @param self: self sent to method

    Pass if amount_field.validate returns False
    """

    def test_written_amount_too_large(self):
        print("\tAmount Validation Test 6")
        test_amount_field_class = AmountField
        test_pair = field_data.FieldData

        test_pair.field_type = field_data.FieldType.FIELD_TYPE_AMOUNT
        test_pair.extracted_data = "One million"
        self.assertEqual(test_amount_field_class.validate(AmountField, test_pair), False)

    """
    Tests if string of written numbers is sent that are valid, should pass

    @param self: self sent to method

    Pass if amount_field.validate returns True
    """
    def test_written_amount_valid(self):
        print("\tAmount Validation Test 7")
        test_amount_field_class = AmountField
        test_pair = field_data.FieldData

        test_pair.field_type = field_data.FieldType.FIELD_TYPE_AMOUNT
        test_pair.extracted_data = "One hundred and fifty"
        self.assertEqual(test_amount_field_class.validate(AmountField, test_pair), True)

if __name__ == '__main__':
    unittest.main()
