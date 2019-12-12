from .data.field_data import FieldData, FieldType
from .field import Field

"""
This class represents the Memo Field, which is the field containing the information about the memo of the check

It extends the Field class
* Has a validate method: to validate the date that is extracted from the check
* Has get_type method: to return its' type
"""


class MemoField(Field):
    """
    Validates the Memo that is extracted from the check

    @param self: self sent to method
    @param data: FieldData class that contains the extracted_data (the Memo on the check)

    @return True if extracted Memo was valid (it exists), False otherwise
    """
    def validate(self, data: FieldData):
        print("Validating if the passed data is a valid memo number")

        print("The memo is valid.")
        return True

    """
    Gets the field type of this class

    @param self: self sent to method

    @return FIELD_TYPE_MEMO the enum Field type of the class
    """
    def get_type(self):
        return FieldType.FIELD_TYPE_MEMO
