import backend.data_extraction.field.data.field_data as field_data
import backend.data_extraction.field.field as field
import datetime

class DateField(field.Field):

    def validate(self, data: field_data.FieldData):
        print("Validating if the passed data is a valid date")
        date = data.extracted_data
        try:
            day, month, year = date.split('/')
        except ValueError:
            print("The date is invalid")
            return False

        try:
            datetime.datetime(int(year), int(month), int(day))
        except ValueError:
            print("The date is invalid.")
            return False

        print("The date is valid.")
        return True

      
    def get_type(self):
        return field_data.FieldType.FIELD_TYPE_DATE