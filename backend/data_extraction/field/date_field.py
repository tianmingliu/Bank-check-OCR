import backend.data_extraction.field.data.field_data as field_data
import backend.data_extraction.field.field as field

class DateField(field.Field):
    def identify(self, field_data: field_data.DataPair):
        print("Identifying if the passed data is the date...")

    def validate(self, data: field_data.DataPair):
        print("Validating if the passed data is a valid date")
    
    def getType(self):
        return field_data.FieldType.FIELD_TYPE_DATE