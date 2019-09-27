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

@return A list of FieldData containg the bounding boxes of fields
extracted from the image.
"""
def extractFieldsEntryPoint(image):
    print("Extracting fields...")
    # do stuff
    ret, thresh = cv2.threshold(image,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    cv2.imshow("Threshold detection", thresh)
    cv2.waitKey(0)

    # Destroying present windows on screen 
    cv2.destroyAllWindows()  

    # noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 1)

    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=5)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)

    cv2.imshow("Sure foreground", sure_fg)
    cv2.waitKey(0)

    # Destroying present windows on screen 
    cv2.destroyAllWindows()  

    cv2.imshow("Sure background", sure_bg)
    cv2.waitKey(0)

    # Destroying present windows on screen 
    cv2.destroyAllWindows()  

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1

    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0

    #markers = cv2.watershed(image,markers)
    #image[markers == -1] = [255,0,0]

    #cv2.imshow("JET Colormap", image)
    #cv2.waitKey(0)

    # Destroying present windows on screen 
    #cv2.destroyAllWindows()  

    # Just return a single data field for now
    list = [ field_data.FieldData(), field_data.FieldData() ]

    # returns the data struct
    return list