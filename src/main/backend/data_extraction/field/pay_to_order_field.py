from .data.field_data import FieldData, FieldType
from .field import Field

class PayToOrderField(Field):

    def validate(self, data: FieldData):
        print("Validating if the passed data is a valid pay to the order of data")
        extracted = str(data.extracted_data)

        if any(c.isalpha() for c in extracted):
            return True
        else:
            return False

    def get_type(self):
        return FieldType.FIELD_TYPE_PAY_TO_ORDER
