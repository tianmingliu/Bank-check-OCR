import backend.data_extraction.field.data.field_data as field_data
import backend.data_extraction.field.field as field


class RoutingNumberField(field.Field):
    validRoutingNumbers = [91408501, 91300023, 92900383, 122105155, 83900363,
                           102000021, 307070115, 42000013, 91000022, 101200453,
                           121201694, 122235821, 123000220, 107002312, 75000022,
                           73000545, 71904779, 42100175, 101000187, 81202759,
                           124302150, 82000549, 121122676, 91300023, 125000105,
                           74900783, 41202582, 104000029, 102101645, 91215927,
                           81000210, 64000059, 104000029, 123103729, 91000022
                           ]  # Valid routing numbers of the different states

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

        else:
            for validNum in self.validRoutingNumbers:
                if num_routing == validNum:
                    print("This is a valid account number.")
                    return True

            print("This is not a valid account number.")
            return False

    def get_type(self):
        return field_data.FieldType.FIELD_TYPE_ROUTING
