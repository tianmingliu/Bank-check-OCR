import unittest

from src.main.backend.data_extraction.field.account_number_field import AccountNumberField
from src.main.backend.data_extraction.field.data.field_data import FieldData, FieldType

"""
This class test the AccountNumberField class
It extends the unittest TestCase class
"""
class AccountNumberTest(unittest.TestCase):

    """
    Tests if string of numbers is sent, there should be no problems parsing it

    @param self: self sent to method

    Pass if account_number_field.validate returns True
    """
    def test_is_float(self):
        print("\tAccount Validation Test 1")
        test_account_number_field_class = AccountNumberField()
        test_pair = FieldData()

        test_pair.field_type = FieldType.FIELD_TYPE_ACCOUNT
        test_pair.extracted_data = "12345678"
        self.assertEqual(test_account_number_field_class.validate(test_pair), True)

    """
    Tests if string of letters is sent, should fail

    @param self: self sent to method

    Pass if account_number_field.validate returns False
    """
    def test_is_not_float(self):
        print("\tAccount Validation Test 2")
        test_account_number_field_class = AccountNumberField()
        test_pair = FieldData()

        test_pair.field_type = FieldType.FIELD_TYPE_ACCOUNT
        test_pair.extracted_data = "abcdefghijk"
        self.assertEqual(test_account_number_field_class.validate(test_pair), False)

    """
    Tests if string of numbers too small is sent, should fail

    @param self: self sent to method

    Pass if account_number_field.validate returns False
    """
    def test_is_too_small(self):
        print("\tAccount Validation Test 3")
        test_account_number_field_class = AccountNumberField()
        test_pair = FieldData()

        test_pair.field_type = FieldType.FIELD_TYPE_ACCOUNT
        test_pair.extracted_data = "123"
        self.assertEqual(test_account_number_field_class.validate(test_pair), False)

    """
    Tests if string of numbers too large is sent, should fail

    @param self: self sent to method

    Pass if account_number_field.validate returns False
    """
    def test_is_too_big(self):
        print("\tAccount Validation Test 4")
        test_account_number_field_class = AccountNumberField()
        test_pair = FieldData()

        test_pair.field_type = FieldType.FIELD_TYPE_ACCOUNT
        test_pair.extracted_data = "1234567891011121314"
        self.assertEqual(test_account_number_field_class.validate(test_pair), False)

    """
    Tests valid string of numbers, should pass

    @param self: self sent to method

    Pass if account_number_field.validate returns True
    """
    def test_passes_everything(self):
        print("\tAccount Validation Test 5")
        test_account_number_field_class = AccountNumberField()
        test_pair = FieldData()

        test_pair.field_type = FieldType.FIELD_TYPE_ACCOUNT
        test_pair.extracted_data = "123456789"
        self.assertEqual(test_account_number_field_class.validate(test_pair), True)


if __name__ == '__main__':
    unittest.main()
