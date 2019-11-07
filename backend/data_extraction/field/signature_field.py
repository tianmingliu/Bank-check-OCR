import backend.data_extraction.field.data.field_data as field_data
import backend.data_extraction.field.field as field

class SignatureField(field.Field):
    def identify(self, field_data: field_data.DataPair):
        print("Identifying if the passed data is the signature...")

    def validate(self, data: field_data.DataPair):
        print("Validating if the passed data is a valid signature")
        extracted = str(data.data.extracted_data)
        if extracted.strip() == "":
            return False
        else:
            return True

    def get_type(self):
        return field_data.FieldType.FIELD_TYPE_SIGNATURE