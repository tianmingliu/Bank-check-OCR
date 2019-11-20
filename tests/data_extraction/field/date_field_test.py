import unittest

from backend.data_extraction.field.date_field import DateField
from backend.data_extraction.field.data import field_data

"""
This class tests the DateField class
It extends the unittest TestCase class
"""


class MyTestCase(unittest.TestCase):

    """
    Tests if a date is sent without slashes, it should fail since it can't be split

    @param self: self sent to method

    Pass if date_field.validate returns False
    """
    def test_date_cannot_split_no_slash(self):
        print("\tDate Validation Test 1")
        test_date_field_class = DateField
        test_pair = field_data.FieldData

        test_pair.field_type = field_data.FieldType.FIELD_TYPE_DATE
        test_pair.extracted_data = "12-10-19"
        self.assertEqual(test_date_field_class.validate(DateField, test_pair), False)

    """
    Tests if a date is sent with only 1 slash, it should fail since it can't be split

    @param self: self sent to method

    Pass if date_field.validate returns False
    """
    def test_date_cannot_split_one_slash(self):
        print("\tDate Validation Test 2")
        test_date_field_class = DateField
        test_pair = field_data.FieldData

        test_pair.field_type = field_data.FieldType.FIELD_TYPE_DATE
        test_pair.extracted_data = "12/10-19"
        self.assertEqual(test_date_field_class.validate(DateField, test_pair), False)

    """
    Tests if a date is sent with an inaccurate day (too large or too small)

    @param self: self sent to method

    Pass if date_field.validate returns False
    """
    def test_date_datetime_bad_day(self):
        print("\tDate Validation Test 3")
        test_date_field_class = DateField
        test_pair = field_data.FieldData

        test_pair.field_type = field_data.FieldType.FIELD_TYPE_DATE
        test_pair.extracted_data = "32/10/19"
        self.assertEqual(test_date_field_class.validate(DateField, test_pair), False)

    """
    Tests if a date is sent with an inaccurate month (too large or too small)

    @param self: self sent to method

    Pass if date_field.validate returns False
    """
    def test_date_datetime_bad_month(self):
        print("\tDate Validation Test 4")
        test_date_field_class = DateField
        test_pair = field_data.FieldData

        test_pair.field_type = field_data.FieldType.FIELD_TYPE_DATE
        test_pair.extracted_data = "12/13/19"
        self.assertEqual(test_date_field_class.validate(DateField, test_pair), False)

    """
    Tests if a date is sent with an inaccurate year (0 or greater than 4 digits)
    
    @param self: self sent to method
    
    Pass if date_field.validate returns False
    """
    def test_date_datetime_bad_year(self):
        print("\tDate Validation Test 5")
        test_date_field_class = DateField
        test_pair = field_data.FieldData

        test_pair.field_type = field_data.FieldType.FIELD_TYPE_DATE
        test_pair.extracted_data = "12/12/0"        # only works with single digits > 1, or any 2, 3, or 4 digit number.
        self.assertEqual(test_date_field_class.validate(DateField, test_pair), False)

    """
    Tests if a valid date is sent, it should pass

    @param self: self sent to method

    Pass if date_field.validate returns True
    """
    def test_date_valid(self):
        print("\tDate Validation Test 6")
        test_date_field_class = DateField
        test_pair = field_data.FieldData

        test_pair.field_type = field_data.FieldType.FIELD_TYPE_DATE
        test_pair.extracted_data = "12/12/19"
        self.assertEqual(test_date_field_class.validate(DateField, test_pair), True)

if __name__ == '__main__':
    unittest.main()
