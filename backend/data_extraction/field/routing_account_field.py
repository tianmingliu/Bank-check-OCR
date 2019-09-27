import backend.data_extraction.field.data.field_data as field_data
import backend.data_extraction.field.field as field

class RoutingField(field.Field):
    def identify(self, field_data: field_data.FieldData):
        print("Identifying if the passed data is the routing number...")

    def validate(self, data: field_data.FieldData):
        print("Validating if the passed data is a valid routing number")

    def getType(self):
        return field_data.FieldType.FIELD_TYPE_ROUTING