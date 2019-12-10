from .data.field_data import FieldData, FieldType
from .field import Field
import datetime
from dateutil.parser import parse
from dateutil import relativedelta
from ..config_helper import get_config_data

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
            parsed_date = parse(date).date()
            print(parsed_date)
        except ValueError:
            print("Invalid date format")
            return False

        today = datetime.date.today()
        months_config = int(get_config_data()['thresholds']['check_age_limit_months'])
        age_limit = today - relativedelta.relativedelta(months=months_config)

        if parsed_date > today:
            print("Date is after today")
            return False
        elif parsed_date < age_limit:
            print(age_limit)
            print("Check is too old")
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
