import backend.data_extraction.field.data.field_data as field_data
import backend.data_extraction.field.field as field

class MemoField(field.Field):

    def validate(self, data: field_data.FieldData):
        print("Validating if the passed data is a valid memo number")

        try:
            str(data.extracted_data)
        except ValueError:
            print("Memo is not a string")
            return False

        print("The memo is valid.")
        return True


    def get_type(self):
        return field_data.FieldType.FIELD_TYPE_MEMO