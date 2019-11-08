import backend.data_extraction.field.data.field_data as field_data
import backend.data_extraction.field.field as field

class RoutingNumberField(field.Field):

    def validate(self, data: field_data.FieldData):
        print("Validating if the passed data is a valid routing number")
        try:
            num_routing = str(data.extracted_data)
        except ValueError:
            print("Routing number is not a float")
            return False

        if not num_routing.isdigit():
            print("This is not a valid routing number.")
            return False

        elif len(num_routing) != 9:
            print("This is not a valid routing number.")
            return False

        else:
            print("This is a valid routing number.")
            return True

    def get_type(self):
        return field_data.FieldType.FIELD_TYPE_ROUTING