import backend.data_extraction.field.field                 as field
import backend.data_extraction.field.amount_field          as amount_field
import backend.data_extraction.field.date_field            as date_field
import backend.data_extraction.field.routing_number_field as routing_number_field
import backend.data_extraction.field.account_number_field as account_number_field
import backend.data_extraction.field.signature_field       as signature_field
import backend.data_extraction.field.data.field_data       as field_data
import backend.data_extraction.field.pay_to_order_field as pay_to_order


# NOTE(Dustin): Maybe use an ordered map instead? Have the key
# be the FieldType and the data be the object. Ordered maps tend to
# be optmized for iteration, but also allow for fast lookups. Iterate
# over the map when identifying and lookup when validating.
# Dictionary of Field classes.
#    Key: Type of FieldData
#    Value: Class that has implemented the Field interface.
GlobalFieldList = {
    field_data.FieldType.FIELD_TYPE_AMOUNT:    amount_field.AmountField(),
    field_data.FieldType.FIELD_TYPE_AMOUNT_WRITTEN: amount_field.AmountField(),
    field_data.FieldType.FIELD_TYPE_DATE:      date_field.DateField(),
    field_data.FieldType.FIELD_TYPE_ROUTING:   routing_number_field.RoutingNumberField(),
    field_data.FieldType.FIELD_TYPE_ACCOUNT:   account_number_field.AccountNumberField(),
    field_data.FieldType.FIELD_TYPE_SIGNATURE: signature_field.SignatureField(),
    field_data.FieldType.FIELD_TYPE_PAY_TO_ORDER_OF: pay_to_order.PayToOrderField(),
}