import data_extraction.field.data.field_data as field_data
import data_extraction.field.field as field

class DateField(field.Field):
    def identify(self, field_data: field_data.FieldData):
        print("Identifying if the passed data is the date...")

    def validate(self, data: field_data.FieldData):
        print("Validating if the passed data is a valid date")
    
    def getType(self):
        return field_data.FieldType.FIELD_TYPE_DATE