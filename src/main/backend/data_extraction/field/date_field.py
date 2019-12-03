from .data.field_data import FieldData, FieldType
from .field import Field
import datetime

"""
This class represents the Date Field, which is the field containing the information about the date of the check

It extends the Field class
* Has a validate method: to validate the date that is extracted from the check
* Has get_type method: to return its' type
"""


class DateField(Field):

    """
    Validates the date that is extracted from the check

    @param self: self sent to method
    @param data: FieldData class that contains the extracted_data (the date on the check)

    @return True if extracted date was valid, False otherwise
    """
    def validate(self, data: FieldData):
        print("Validating if the passed data is a valid date")
        date = data.extracted_data
        print(date)
        try:
            month, day, year = date.split('/')
        except ValueError:
            try:
                month, day, year = date.split('-')
            except ValueError:
                print("The date is invalid (cannot be split).")
                return False

        try:
            print(year + " " + month + " " + day)
            datetime.datetime(int(year), int(month), int(day))
        except ValueError:
            print("The date is invalid (Date time issue).")
            return False

        print("The date is valid.")
        return True

    """
    Gets the field type of this class

    @param self: self sent to method

    @return FIELD_TYPE_DATE the enum Field type of the class
    """
    def get_type(self):
        return FieldType.FIELD_TYPE_DATE
