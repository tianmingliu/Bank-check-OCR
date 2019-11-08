import backend.data_extraction.field.data.field_data as field_data
import backend.data_extraction.field.field as field

class DateField(field.Field):

    def validate(self, data: field_data.FieldData):
        print("Validating if the passed data is a valid date")
        return False
    
    def get_type(self):
        return field_data.FieldType.FIELD_TYPE_DATE