from .field.amount_field import AmountField
from .field.date_field import DateField
from .field.routing_number_field import RoutingNumberField
from .field.account_number_field import AccountNumberField
from .field.signature_field import SignatureField
from .field.pay_to_order_field import PayToOrderField
from .field.memo_field import MemoField
from .field.data.field_data import FieldType


# NOTE(Dustin): Maybe use an ordered map instead? Have the key
# be the FieldType and the data be the object. Ordered maps tend to
# be optmized for iteration, but also allow for fast lookups. Iterate
# over the map when identifying and lookup when validating.
# Dictionary of Field classes.
#    Key: Type of FieldData
#    Value: Class that has implemented the Field interface.
GlobalFieldList = {
    FieldType.FIELD_TYPE_AMOUNT:    AmountField(),
    FieldType.FIELD_TYPE_AMOUNT_WRITTEN: AmountField(),
    FieldType.FIELD_TYPE_DATE:      DateField(),
    FieldType.FIELD_TYPE_ROUTING:   RoutingNumberField(),
    FieldType.FIELD_TYPE_ACCOUNT:   AccountNumberField(),
    FieldType.FIELD_TYPE_SIGNATURE: SignatureField(),
    FieldType.FIELD_TYPE_PAY_TO_ORDER_OF: PayToOrderField(),
    FieldType.FIELD_TYPE_MEMO: MemoField(),
}