import unittest
import datetime
from dateutil import relativedelta
from src.main.backend.data_extraction.field.date_field import DateField
from src.main.backend.data_extraction.field.data.field_data import FieldData, FieldType

"""
This class test the DateField class
It extends the unittest TestCase class
"""


class DateTest(unittest.TestCase):

    """
    Tests if a date is sent without slashes, it should fail since it can't be split

    @param self: self sent to method

    Pass if date_field.validate returns False
    """
    def test_date_cannot_split_no_slash(self):
        print("\tDate Validation Test 1")
        test_date_field_class = DateField()
        test_pair = FieldData()

        test_pair.field_type = FieldType.FIELD_TYPE_DATE
        test_pair.extracted_data = "121019"
        self.assertEqual(test_date_field_class.validate(test_pair), False)

    """
    Tests if a date is sent with only 1 slash, it should fail since it can't be split

    @param self: self sent to method

    Pass if date_field.validate returns False
    """
    def test_date_cannot_split_one_slash(self):
        print("\tDate Validation Test 2")
        test_date_field_class = DateField()
        test_pair = FieldData()

        test_pair.field_type = FieldType.FIELD_TYPE_DATE
        test_pair.extracted_data = "12/10-19"
        self.assertEqual(test_date_field_class.validate(test_pair), False)

    """
    Tests if a date is sent with an inaccurate day (too large or too small)

    @param self: self sent to method

    Pass if date_field.validate returns False
    """
    def test_date_datetime_bad_day(self):
        print("\tDate Validation Test 3")
        test_date_field_class = DateField()
        test_pair = FieldData()

        test_pair.field_type = FieldType.FIELD_TYPE_DATE
        test_pair.extracted_data = "32/10/19"
        self.assertEqual(test_date_field_class.validate(test_pair), False)

    """
    Tests if a date is sent with an inaccurate month (too large or too small)

    @param self: self sent to method

    Pass if date_field.validate returns False
    """
    def test_date_datetime_bad_month(self):
        print("\tDate Validation Test 4")
        test_date_field_class = DateField()
        test_pair = FieldData()

        test_pair.field_type = FieldType.FIELD_TYPE_DATE
        test_pair.extracted_data = "13/12/19"
        self.assertEqual(test_date_field_class.validate(test_pair), False)

    """
    Tests if a date is sent with an inaccurate year (0 or greater than 4 digits)
    
    @param self: self sent to method
    
    Pass if date_field.validate returns False
    """
    def test_date_datetime_bad_year(self):
        print("\tDate Validation Test 5")
        test_date_field_class = DateField()
        test_pair = FieldData()

        test_pair.field_type = FieldType.FIELD_TYPE_DATE
        test_pair.extracted_data = "12/12/0"        # only works with single digits > 1, or any 2, 3, or 4 digit number.
        self.assertEqual(test_date_field_class.validate(test_pair), False)

    """
    Tests if a valid date is sent, it should pass

    @param self: self sent to method

    Pass if date_field.validate returns True
    """
    def test_date_valid(self):
        print("\tDate Validation Test 6")
        test_date_field_class = DateField()
        test_pair = FieldData()

        test_pair.field_type = FieldType.FIELD_TYPE_DATE
        today = datetime.date.today()
        date = today - relativedelta.relativedelta(months=3)
        test_pair.extracted_data = str(date.month) + "/" + str(date.day) + "/" + str(date.year)
        self.assertEqual(test_date_field_class.validate(test_pair), True)

    """
    Tests if a date that is older than 6 months, it should fail

    @param self: self sent to method

    Pass if date_field.validate returns False
    """
    def test_date_old(self):
        test_date_field_class = DateField()
        test_pair = FieldData()

        test_pair.field_type = FieldType.FIELD_TYPE_DATE

        today = datetime.date.today()
        date = today - relativedelta.relativedelta(months=8)
        test_pair.extracted_data = str(date.month) + "/" + str(date.day) + "/" + str(date.year)

        self.assertEqual(test_date_field_class.validate(test_pair), False)

    """
    Tests if a date that is in the future, it should fail

    @param self: self sent to method

    Pass if date_field.validate returns False
    """
    def test_date_future(self):
        test_date_field_class = DateField()
        test_pair = FieldData()

        test_pair.field_type = FieldType.FIELD_TYPE_DATE

        today = datetime.date.today()
        date = today + relativedelta.relativedelta(months=1)
        test_pair.extracted_data = str(date.month) + "/" + str(date.day) + "/" + str(date.year)

        self.assertEqual(test_date_field_class.validate(test_pair), False)


if __name__ == '__main__':
    unittest.main()
