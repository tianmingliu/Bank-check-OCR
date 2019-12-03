from .data.field_data import FieldData, FieldType
from .field import Field


class SignatureField(Field):

    def validate(self, data: FieldData):
        print("Validating if the passed data is a valid signature")
        extracted = str(data.extracted_data)
        if extracted.strip() == "":
            return False
        else:
            return True

    def get_type(self):
        return FieldType.FIELD_TYPE_SIGNATURE
