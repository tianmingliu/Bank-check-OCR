import unittest

from backend.data_extraction.field.routing_number_field import RoutingNumberField
from backend.data_extraction.field.data import field_data

"""
This class tests the RoutingNumberField class
It extends the unittest TestCase class
"""


class MyTestCase(unittest.TestCase):

    """
    Tests if string of letters is sent, should fail

    @param self: self sent to method

    Pass if routing_number_field.validate returns False
    """
    def test_is_not_float(self):
        print("\tRouting Validation Test 1")
        test_routing_number_field_class = RoutingNumberField
        test_pair = field_data.FieldData

        test_pair.field_type = field_data.FieldType.FIELD_TYPE_ROUTING
        test_pair.extracted_data = "abcdefghijk"
        self.assertEqual(test_routing_number_field_class.validate(RoutingNumberField, test_pair), False)

    """
    Tests if valid (in the array of valid routing numbers) string of digits is sent, should pass

    @param self: self sent to method

    Pass if routing_number_field.validate returns True
    """
    def test_is_valid(self):
        print("\tRouting Validation Test 2")
        test_routing_number_field_class = RoutingNumberField
        test_pair = field_data.FieldData

        test_pair.field_type = field_data.FieldType.FIELD_TYPE_ROUTING
        test_pair.extracted_data = "91408501"
        self.assertEqual(test_routing_number_field_class.validate(RoutingNumberField, test_pair), True)

    """
    Tests if not valid (not in the array of valid routing numbers) string of digits is sent, should fail

    @param self: self sent to method

    Pass if routing_number_field.validate returns False
    """
    def test_is_not_valid(self):
        print("\tRouting Validation Test 3")
        test_routing_number_field_class = RoutingNumberField
        test_pair = field_data.FieldData

        test_pair.field_type = field_data.FieldType.FIELD_TYPE_ROUTING
        test_pair.extracted_data = "12345678"
        self.assertEqual(test_routing_number_field_class.validate(RoutingNumberField, test_pair), False)

if __name__ == '__main__':
    unittest.main()
