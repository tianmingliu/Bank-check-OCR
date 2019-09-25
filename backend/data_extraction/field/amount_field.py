"""
There are two types of amounts on a check:
1. $250.00
2. Two hundred and fifty dollars

There are two versions then to validate. The passed data will accept a 
written and a non-written field for the identify method.
"""
import backend.data_extraction.field.data.field_data as field_data
import backend.data_extraction.field.field as field
import backend.data_extraction.config_helper as helper


class AmountField(field.Field):

    def identify(self, field_data: field_data.FieldData):
        print("Identifying if the passed data is the amount...")

    def validate(self, data: field_data.FieldData):
        print("Validating if the passed data is a valid amount")

        try:
            num_amount = float(data.data_info.extractedData)
        except ValueError:
            print("Amount number is not a float")
            return False

        if num_amount < helper.get_config_data()['thresholds']['amount_min']:
            print("Amount number is lower than minimum")
            return False
        elif num_amount > helper.get_config_data()['thresholds']['amount_max']:
            print("Amount number is greater than maximum")
            return False
        else:
            print("Passed validation")
            return True

    def getType(self):
        return field_data.FieldType.FIELD_TYPE_AMOUNT

