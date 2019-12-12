from .data.field_data import FieldData, FieldType
from .field import Field

"""
This class represents the Pay to the Order Of Field, which is the field containing the information about who the check
is written for

It extends the Field class
* Has a validate method: to validate the date that is extracted from the check
* Has get_type method: to return its' type
"""


class PayToOrderField(Field):
    """
    Validates the Pay to the Order Of data that is extracted from the check

    @param self: self sent to method
    @param data: FieldData class that contains the extracted_data (the Pay to the Order Of on the check)

    @return True if extracted 'Pay to the Order Of' was valid, False otherwise
    """
    def validate(self, data: FieldData):
        print("Validating if the passed data is a valid pay to the order of data")
        extracted = str(data.extracted_data)

        if any(c.isalpha() for c in extracted):
            return True
        else:
            return False

    """
    Gets the field type of this class

    @param self: self sent to method

    @return FIELD_TYPE_PAY_TO_ORDER the enum Field type of the class
    """
    def get_type(self):
        return FieldType.FIELD_TYPE_PAY_TO_ORDER
