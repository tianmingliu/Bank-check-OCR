import backend.data_extraction.field.field                 as field
import backend.data_extraction.field.amount_field          as amount_field
import backend.data_extraction.field.date_field            as date_field
import backend.data_extraction.field.routing_account_field as routing_account_field
import backend.data_extraction.field.signature_field       as signature_field
import backend.data_extraction.field.data.field_data       as field_data


# NOTE(Dustin): Maybe use an ordered map instead? Have the key
# be the FieldType and the data be the object. Ordered maps tend to
# be optmized for iteration, but also allow for fast lookups. Iterate
# over the map when identifying and lookup when validating.
# Dictionary of Field classes.
#    Key: Type of FieldData
#    Value: Class that has implemented the Field interface.
GlobalFieldList = {
    field_data.FieldType.FIELD_TYPE_AMOUNT:    amount_field.AmountField(),
    field_data.FieldType.FIELD_TYPE_DATE:      date_field.DateField(),
    field_data.FieldType.FIELD_TYPE_ROUTING:   routing_account_field.RoutingField(),
    field_data.FieldType.FIELD_TYPE_SIGNATURE: signature_field.SignatureField(),
}