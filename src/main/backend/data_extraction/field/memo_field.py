from .data.field_data import FieldData, FieldType
from .field import Field

class MemoField(Field):

    def validate(self, data: FieldData):
        print("Validating if the passed data is a valid memo number")

        print("The memo is valid.")
        return True


    def get_type(self):
        return FieldType.FIELD_TYPE_MEMO
