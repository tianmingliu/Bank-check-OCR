import backend.data_extraction.field.data.field_data as field_data
import backend.data_extraction.field.field as field

class MemoField(field.Field):
    def identify(self, field: field_data.DataPair):
        print("Identifying if the passed data is the memo number...")

        if field_data.DataPair.data.field_type == field.field_data.FieldType.FIELD_TYPE_MEMO:
            return True
        else:
            return False

    def validate(self, data: field_data.DataPair):
        print("Validating if the passed data is a valid memo number")

        try:
            str(data.data_info.extractedData)
        except ValueError:
            print("Memo is not a string")
            return False

        print("The memo is valid.")
        return True


    def get_type(self):
        return field_data.FieldType.FIELD_TYPE_MEMO