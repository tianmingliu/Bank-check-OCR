import backend.data_extraction.field.data.field_data as field_data
import backend.data_extraction.field.field as field

class AccountNumberField(field.Field):
    def identify(self, field_data: field_data.DataPair):
        print("Identifying if the passed data is the account number...")

        if field_data.DataPair.data.field_type == field.field_data.FieldType.FIELD_TYPE_ACCOUNT:
            return True
        else:
            return False

    def validate(self, data: field_data.DataPair):
        print("Validating if the passed data is a valid account number")
        try:
            num_account = float(data.data_info.extractedData)
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