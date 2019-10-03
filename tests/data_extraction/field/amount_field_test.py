import unittest
import backend.data_extraction.field.data.field_data as field_data
import backend.data_extraction.field.amount_field as amount

class AmountFieldTest(unittest.TestCase):
    def test_validate(self):
        test_amount = amount.AmountField()
        test_data1 = field_data.FieldData(
            field_data.FieldType.FIELD_TYPE_AMOUNT,
            field_data.BoundingBox(0.0, 0.0),
            field_data.FieldDataInfo("1000.50")
        )

        self.assertEqual(test_amount.validate(test_data1), True)

        test_data1.data_info.extractedData = "-100"

        self.assertEqual(test_amount.validate(test_data1), False)

        test_data1.data_info.extractedData = "200000000"

        self.assertEqual(test_amount.validate(test_data1), False)

        test_data1.data_info.extractedData = "100O"

        self.assertEqual(test_amount.validate(test_data1), False)

if __name__ == '__main__':
    unittest.main()
