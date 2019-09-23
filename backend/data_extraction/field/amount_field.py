"""
There are two types of amounts on a check:
1. $250.00
2. Two hundred and fifty dollars

There are two versions then to validate. The passed data will accept a 
written and a non-written field for the identify method.
"""
import data_extraction.field.data.field_data as field_data
import data_extraction.field.field as field

class AmountField(field.Field):
    def identify(self, field_data: field_data.FieldData):
        print("Identifying if the passed data is the amount...")

    def validate(self, data: field_data.FieldData):
        print("Validating if the passed data is a valid amount")

    def getType(self):
        return field_data.FieldType.FIELD_TYPE_AMOUNT
