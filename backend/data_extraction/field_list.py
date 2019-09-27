import backend.data_extraction.field.field                 as field
import backend.data_extraction.field.amount_field          as amount_field
import backend.data_extraction.field.date_field            as date_field
import backend.data_extraction.field.routing_account_field as routing_account_field
import backend.data_extraction.field.signature_field       as signature_field

# NOTE(Dustin): Maybe use an ordered map instead? Have the key
# be the FieldType and the data be the object. Ordered maps tend to
# be optmized for iteration, but also allow for fast lookups. Iterate
# over the map when identifying and lookup when validating.
GlobalFieldList = [
    amount_field.AmountField(),
    date_field.DateField(),
    routing_account_field.RoutingField(),
    signature_field.SignatureField(),
]