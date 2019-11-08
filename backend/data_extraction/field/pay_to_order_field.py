import backend.data_extraction.field.data.field_data as field_data
import backend.data_extraction.field.field as field

class PayToOrderField(field.Field):

    def validate(self, data: field_data.FieldData):
        print("Validating if the passed data is a valid pay to the order of data")
        extracted = str(data.extracted_data)

        if any(c.isalpha() for c in extracted):
            return True
        else:
            return False

    def get_type(self):
        return field_data.FieldType.FIELD_TYPE_PAY_TO_ORDER
