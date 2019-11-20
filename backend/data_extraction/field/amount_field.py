"""
There are two types of amounts on a check:
1. $250.00
2. Two hundred and fifty dollars

There are two versions then to validate. The passed data will accept a 
written and a non-written field for the identify method.
"""
import backend.data_extraction.field.data.field_data as field_data
import backend.data_extraction.field.field           as field
import backend.data_extraction.config_helper         as helper


class AmountField(field.Field):

    """
    Validate the amount field provided. Checks if the value is between the configurable
    max and min and also ensures the data provided is actually a numerical amount.
    Returns false when failing validation and true if pass.
    """
    def validate(self, data: field_data.FieldData):
        print("Validating if the passed data is a valid amount")

        try:
            num_amount = float(data.extracted_data)
        except ValueError:
            print("Amount number is not a float")

            written_amount = str(data.extracted_data)

            try:
                num_amount = word_to_num_helper(written_amount)
            except ValueError:
                print("Invalid written amount")
                return False

        if num_amount <= helper.get_config_data()['thresholds']['amount_min']:
            print("Amount number is lower than minimum: " + str(helper.get_config_data()['thresholds']['amount_min']))
            return False
        elif num_amount >= helper.get_config_data()['thresholds']['amount_max']:
            print("Amount number is greater than maximum: " + str(helper.get_config_data()['thresholds']['amount_max']))
            return False
        else:
            print("Passed validation")
            return True

    def get_type(self):
        return field_data.FieldType.FIELD_TYPE_AMOUNT


"""
Converts written numbers into digit numbers

@param textnum: string of written numbers to be converted
@param numwords: keywords array used for conversion

@return result + current: the final int value that is the converted value of the written amount
"""


def word_to_num_helper(textnum, numwords={}):
    if not numwords:
        units = [
            "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
            "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
            "sixteen", "seventeen", "eighteen", "nineteen",
        ]

        tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

        scales = ["hundred", "thousand", "million", "billion", "trillion"]

        numwords["and"] = (1, 0)
        for idx, word in enumerate(units):
            numwords[word] = (1, idx)
        for idx, word in enumerate(tens):
            numwords[word] = (1, idx * 10)
        for idx, word in enumerate(scales):
            numwords[word] = (10 ** (idx * 3 or 2), 0)

    current = result = 0
    textnum = textnum.replace('-', ' ')
    textnum = textnum.lower()

    for word in textnum.split():
        if word not in numwords:
            raise ValueError("Illegal word: " + word)

        scale, increment = numwords[word]
        current = current * scale + increment
        if scale > 100:
            result += current
            current = 0

    return result + current
