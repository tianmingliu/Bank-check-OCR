import cv2

from backend.data_extraction.field.data.field_data import FieldData
from backend.data_extraction.field.data.field_data import BoundingRect

def postprocessEntryPoint(image, dim, fields):

    # for (field, _) in fields:
    #     bb : BoundingRect = field.bounds
    #     cv2.rectangle(image, (bb.x, bb.y), (bb.x + bb.w, bb.y + bb.h), (128, 0, 128), 2)




    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    return image