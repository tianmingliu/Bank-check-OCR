import backend.data_extraction.field.data.field_data as field_data

import numpy as np
import cv2

"""
Provided an image of a check, a series of fields are
extracted from the image. For each field found, a FieldData
struct is created. This struct contains default values except
for the bounding box. The bounding box of the field is recorded
for further processing in a later pipelin stage.

@param image: Image of the check to extract fields from

@return A list of DataPair containg the bounding boxes of fields
extracted from the image and the cropped image.
"""
def extractFieldsEntryPoint(image_orig, image):
    print("Extracting fields...")

    _, mask = cv2.threshold(image,180,255,cv2.THRESH_TOZERO_INV)
    image_final = cv2.bitwise_and(image, image, mask=mask)

    cv2.imshow("New image", image_final)
    cv2.waitKey(0)

    # Destroying present windows on screen 
    cv2.destroyAllWindows() 

    _, mask = cv2.threshold(image,180,255,cv2.THRESH_TOZERO_INV)
    _, new_img = cv2.threshold(image_final,150,255,cv2.THRESH_OTSU)  # for black text , cv.THRESH_BINARY_INV

    cv2.imshow("New image", new_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5,
                                                         3))  # to manipulate the orientation of dilution , large x means horizonatally dilating  more, large y means vertically dilating more
    dilated = cv2.dilate(new_img, kernel, iterations=3)  # dilate , more the iteration more the dilation

    cv2.imshow("Dilation", dilated)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 

    # image, contours, hierarchy = cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    contours, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Performs a new dilation on the original image in order to only have black and white
    # ret, cimg = cv2.threshold(image,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    list = []
    img_cpy = image.copy()
    for contour in contours:
        # get rectangle bounding contour
        [x, y, w, h] = cv2.boundingRect(contour)

        # Don't plot small false positives that aren't text
        if w < 35 and h < 35:
            continue

        # draw rectangle around contour on original image
        cv2.rectangle(img_cpy, (x, y), (x + w, y + h), (150, 0, 150), 2)

        # can change to image for only black/white image
        cropped = image_orig[y :y +  h , x : x + w]

        data = field_data.FieldData()
        data.bounds = field_data.BoundingRect(x, y, w, h)
        list.append(field_data.DataPair(cropped, data))

    cv2.imshow('captcha_result', img_cpy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Just return a single data field for now
    # list = [ field_data.FieldData(), field_data.FieldData() ]

    # returns the data struct
    return list