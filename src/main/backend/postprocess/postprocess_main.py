import cv2

"""
Modifies the image to add bounding boxes to it and resize it to original size 

@param image: the image to modify
@param dim: dimensions to draw on to image
@param fields: the fields of the image

@return image: the modified image
"""


def postprocessEntryPoint(image, dim, fields):

    for (field, _) in fields:
        bb : BoundingRect = field.bounds
        cv2.rectangle(image, (bb.x, bb.y), (bb.x + bb.w, bb.y + bb.h), (0, 0, 255), 2)

    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    return image
