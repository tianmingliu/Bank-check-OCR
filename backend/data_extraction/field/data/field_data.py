from enum import Enum
from dataclasses import dataclass
from dataclasses_json import dataclass_json
from typing import List


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
@dataclass_json
class FieldType(Enum):
    FIELD_TYPE_NONE      = 0
    FIELD_TYPE_SIGNATURE = 1
    FIELD_TYPE_DATE      = 2
    FIELD_TYPE_AMOUNT    = 3
    FIELD_TYPE_ROUTING   = 4

"""
Represents a (x,y) coordinate on an image.
"""
@dataclass_json
@dataclass
class Point:
    x: float = 0.0
    y: float = 0.0

"""
Represents a bounding box on an image. 

@field min_bound: Minimum coorindates of the bounding box.
@field max_bound: Maximum coorindates of the bounding box.
"""
@dataclass_json
@dataclass
class BoundingBox:
    min_bound: Point = Point(0.0, 0.0)
    max_bound: Point = Point(0.0, 0.0)

"""
Represents the actual data of a field on a check.

@field extractedData: String representation of the data.
@field confidence: how confident it is that the extracted
                   data is accurate
"""
@dataclass_json
@dataclass
class FieldDataInfo:
    extracted_data: str   = ""
    confidence:    float = 0.0

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
    bounds:     BoundingBox   = BoundingBox()
    data_info:  FieldDataInfo = FieldDataInfo()