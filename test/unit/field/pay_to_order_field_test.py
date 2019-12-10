import unittest

from src.main.backend.data_extraction.field.pay_to_order_field import PayToOrderField
from src.main.backend.data_extraction.field.data.field_data import FieldData, FieldType

"""
This class test the PayToOrderField class
It extends the unittest TestCase class
"""


class PayToOrderTest(unittest.TestCase):

    """
    Tests if string of letter is sent, there should be no problems parsing it

    @param self: self sent to method

    Pass if pay_to_order_field.validate returns True
    """
    def test_is_alpha(self):
        print("\tPay to Order Validation Test 1")
        test_pay_to_order_field_class = PayToOrderField()
        test_pair = FieldData()

        test_pair.field_type = FieldType.FIELD_TYPE_PAY_TO_ORDER_OF
        test_pair.extracted_data = "Tomas"
        self.assertEqual(test_pay_to_order_field_class.validate(test_pair), True)

    """
    Tests if string of number is sent, there should be have problems parsing it

    @param self: self sent to method

    Pass if pay_to_order_field.validate returns False
    """
    def test_is_not_alpha(self):
        print("\tPay to Order Validation Test 2")
        test_pay_to_order_field_class = PayToOrderField()
        test_pair = FieldData()

        test_pair.field_type = FieldType.FIELD_TYPE_PAY_TO_ORDER_OF
        test_pair.extracted_data = "23123123"
        self.assertEqual(test_pay_to_order_field_class.validate(test_pair), False)

    """
    Tests if empty string is sent, there should be have problems parsing it

    @param self: self sent to method

    Pass if pay_to_order_field.validate returns False
    """

    def test_is_empty(self):
        print("\tPay to Order Validation Test 3")
        test_pay_to_order_field_class = PayToOrderField()
        test_pair = FieldData()

        test_pair.field_type = FieldType.FIELD_TYPE_PAY_TO_ORDER_OF
        test_pair.extracted_data = ""
        self.assertEqual(test_pay_to_order_field_class.validate(test_pair), False)



if __name__ == '__main__':
    unittest.main()