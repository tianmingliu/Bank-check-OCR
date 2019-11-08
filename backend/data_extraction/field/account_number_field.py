import backend.data_extraction.field.data.field_data as field_data
import backend.data_extraction.field.field as field

class AccountNumberField(field.Field):

    def validate(self, data: field_data.FieldData):
        print("Validating if the passed data is a valid account number")
        try:
            num_account = str(data.extracted_data)
        except ValueError:
            print("Account number is not a float")
            return False

        if not num_account.isdigit():
            print("This is not a valid account number.")
            return False

        elif len(num_account) < 4 or len(num_account) > 17:
            print("This is not a valid account number.")
            return False

        else:
            print("This is a valid account number.")
            return True

    def get_type(self):
        return field_data.FieldType.FIELD_TYPE_ROUTING
