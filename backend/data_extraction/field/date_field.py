import backend.data_extraction.field.data.field_data as field_data
import backend.data_extraction.field.field as field
import datetime

class DateField(field.Field):
    def identify(self, field_data: field_data.DataPair):
        print("Identifying if the passed data is the date...")

    def validate(self, data: field_data.DataPair):
        print("Validating if the passed data is a valid date")

        date = data.data_info.extractedData
        day, month, year = date.split('/')

        try:
            datetime.datetime(int(year), int(month), int(day))
        except ValueError:
            print("The date is invalid.")
            return False

        print("The date is valid.")
        return True

    
    def get_type(self):
        return field_data.FieldType.FIELD_TYPE_DATE