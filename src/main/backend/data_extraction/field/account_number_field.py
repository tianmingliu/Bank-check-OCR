from .data.field_data import FieldData, FieldType
from .field import Field

"""
This class represents the Account Number Field, which is the field containing the information about the account number
 of the check

It extends the Field class
* Has a validate method: to validate the account number that is extracted from the check
* Has get_type method: to return its' type
"""


class AccountNumberField(Field):
    """
    Validates the account number that is extracted from the check

    @param self: self sent to method
    @param data: FieldData class that contains the extracted_data (the account number on the check)

    @return True if extracted account number was valid, False otherwise
    """
    def validate(self, data: FieldData):
        print("Validating if the passed data is a valid account number")
        try:
            num_account = str(data.extracted_data)
        except ValueError:
            print("Account number can't be cast to a string")
            return False

        if not num_account.isdigit():
            print("This is not a valid account number (not a digit).")
            return False

        elif len(num_account) < 4 or len(num_account) > 17:
            print("This is not a valid account number (too small or large).")
            return False

        else:
            print("This is a valid account number.")
            return True

    """
    Gets the field type of this class

    @param self: self sent to method

    @return FIELD_TYPE_ACCOUNT the enum Field type of the class
    """
    def get_type(self):
        return FieldType.FIELD_TYPE_ACCOUNT
