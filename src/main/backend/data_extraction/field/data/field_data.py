import cv2
from enum             import Enum
from dataclasses      import dataclass
from dataclasses_json import dataclass_json
from typing           import List


"""
Represents a type of field on a check. There are 5 types:
1. NONE
2. Signature
3. Data
4. Amount
5. Routing
NOTE(Dustin): It might be worth splitting date into Written and 
NonWritten
"""
# @dataclass_json

class FieldType(Enum):
    FIELD_TYPE_NONE            = 0
    FIELD_TYPE_DATE            = 1
    FIELD_TYPE_PAY_TO_ORDER_OF = 2
    FIELD_TYPE_AMOUNT          = 3
    FIELD_TYPE_AMOUNT_WRITTEN  = 4
    FIELD_TYPE_SIGNATURE       = 5
    FIELD_TYPE_MEMO            = 6
    FIELD_TYPE_ROUTING         = 7
    FIELD_TYPE_ACCOUNT         = 8


"""
Represents a bounding box on an image. 
@field x: start x coordinate of bounding rectangle
@field y: start y coordinate of bounding rectangle
@field w: width of the bounding rectangle
@field h: height of the bounding rectangle
"""
# @dataclass_json
@dataclass
class BoundingRect:
    x: float = 0
    y: float = 0
    w: float = 0
    h: float = 0

    def __init__(self, x: float, y: float, w: float, h: float):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

        
"""
Represents the data for a Field on a check. 
@field field_type: The type of field represented for the instance.
@field bounds: The bounding box for the field on an image of a check
@field data_info: The information for the field that was extracted from
                  the check
"""
@dataclass_json
@dataclass
class FieldData:
    field_type: FieldType     = FieldType.FIELD_TYPE_NONE
    bounds:     BoundingRect  = BoundingRect(0.0, 0.0, 0.0, 0.0)
    extracted_data: str = ""
    validation: bool = False


"""
Represents the pair of data passed around in the system
@field image: The image that is to be processed
@field data: The data that is extracted from the image
"""
@dataclass
class DataPair:
    image: cv2.IMREAD_GRAYSCALE
    data:  FieldData
