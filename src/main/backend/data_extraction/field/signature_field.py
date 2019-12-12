from .data.field_data import FieldData, FieldType
from .field import Field

"""
This class represents the Signature Field, which is the field containing the signature of the check

It extends the Field class
* Has a validate method: to validate the date that is extracted from the check
* Has get_type method: to return its' type
"""


class SignatureField(Field):
    """
    Validates the Signature data that is extracted from the check

    @param self: self sent to method
    @param data: FieldData class that contains the extracted_data (the Signature on the check)

    @return True if extracted Signature was valid, False otherwise
    """
    def validate(self, data: FieldData):
        print("Validating if the passed data is a valid signature")
        extracted = str(data.extracted_data)
        if extracted.strip() == "":
            return False
        else:
            return True

    """
    Gets the field type of this class

    @param self: self sent to method

    @return FIELD_TYPE_SIGNATURE the enum Field type of the class
    """
    def get_type(self):
        return FieldType.FIELD_TYPE_SIGNATURE
