from .data.field_data import FieldData, FieldType
from .field import Field

class MemoField(Field):

    def validate(self, data: FieldData):
        print("Validating if the passed data is a valid memo number")

        extracted = str(data.extracted_data)

        if extracted.strip() == "":
            print("Memo is empty")
            return False

        print("The memo is valid.")
        return True


    def get_type(self):
        return FieldType.FIELD_TYPE_MEMO
