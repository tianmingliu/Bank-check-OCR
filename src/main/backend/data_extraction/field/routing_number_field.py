from .data.field_data import FieldData, FieldType
from .field import Field

"""
This class represents the Routing Number Field, which is the field containing the information about the routing number
 of the check

It extends the Field class
* Has a validate method: to validate the routing number that is extracted from the check
* Has get_type method: to return its' type
"""


class RoutingNumberField(Field):
    validRoutingNumbers = ["91408501", "91300023", "92900383", "122105155", "83900363",
                           "102000021", "307070115", "42000013", "91000022", "101200453",
                           "121201694", "122235821", "123000220", "107002312", "75000022",
                           "73000545", "71904779", "42100175", "101000187", "81202759",
                           "124302150", "82000549", "121122676", "91300023", "125000105",
                           "74900783", "41202582", "104000029", "102101645", "91215927",
                           "81000210", "64000059", "104000029", "123103729", "91000022"
                           ]  # Valid routing numbers of the different states

    """
    Validates the routing number that is extracted from the check

    @param self: self sent to method
    @param data: FieldData class that contains the extracted_data (the routing number on the check)

    @return True if extracted routing number was valid, False otherwise
    """
    def validate(self, data: FieldData):
        print("Validating if the passed data is a valid routing number")
        try:
            num_routing = str(data.extracted_data)
        except ValueError:
            print("Routing number is not a float")
            return False

        if not num_routing.isdigit():
            print("This is not a valid routing number (not a digit).")
            return False

        else:
            for validNum in self.validRoutingNumbers:
                if num_routing == validNum:
                    print("This is a valid routing number.")
                    return True

            print("This is not a valid routing number (doesn't match from array).")
            return False

    """
    Gets the field type of this class

    @param self: self sent to method

    @return FIELD_TYPE_ROUTING the enum Field type of the class
    """
    def get_type(self):
        return FieldType.FIELD_TYPE_ROUTING
