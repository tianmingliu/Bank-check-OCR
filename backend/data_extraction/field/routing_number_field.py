import backend.data_extraction.field.data.field_data as field_data
import backend.data_extraction.field.field as field

class RoutingNumberField(field.Field):
    def identify(self, field_data: field_data.DataPair):
        print("Identifying if the passed data is the routing number...")

        if field.field_data.DataPair.data.field_type == True:  # temporary condition that should be changed
            return True
        else:
            return False

    def validate(self, data: field_data.DataPair):
        print("Validating if the passed data is a valid routing number")

    def get_type(self):
        return field_data.FieldType.FIELD_TYPE_ROUTING