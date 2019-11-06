"""
Attempted approaches:
- Naive aabb detection
  - the problem with approach is the spacing between characters and lines are inconsistent

- Splitting the image into pieces
  - checks have an inconsistent format for spacing. makes it difficult to, say, pull apart the middle region

- EAST text detection
  - handwriting has inconsentent spacing, which throws the algorithm off. Bounding boxes overlap and are
    inconsistent, and needed a aabb merger. Even then, there are inconsistent results



"""


import backend.data_extraction.field.data.field_data           as field_data

from backend.data_extraction.field.data.field_data import FieldType
from backend.data_extraction.field.data.field_data import FieldData
from backend.data_extraction.field.data.field_data import BoundingRect

import numpy as np
import cv2

from imutils.object_detection import non_max_suppression
import pytesseract
import argparse

east_loc = "resources/east/frozen_east_text_detection.pb"
min_confidence = 0.5 
east_width = 320  # must be multiple of 32
east_height = 320  # must be multiple of 32
# padding_x = 0.1
# padding_y = 0.4

"""
EAST Text detection

Adapted from: https://www.pyimagesearch.com/2018/09/17/opencv-ocr-and-text-recognition-with-tesseract/

"""

"""
A box has the tuple: (startX, startY, endX, endY)
"""
def check_bounding_collision(boxA: tuple, boxB: tuple):
    print("Checking collision...")
    print("    Box A: " + str(boxA))
    print("    Box B: " + str(boxB)) 

    box_sx = boxA[0]
    box_sy = boxA[1]
    box_ex = boxA[2]
    box_ey = boxA[3]

    e_minx = boxB[0]
    e_miny = boxB[1]
    e_maxx = boxB[2]
    e_maxy = boxB[3]

    x_axis = (box_sx <= e_maxx and box_ex >= e_minx)
    y_axis = (box_sy <= e_maxy and box_ey >= e_miny)

    return x_axis and y_axis



def decode_predictions(scores, geometry):
    # grab the number of rows and columns from the scores volume, then
    # initialize our set of bounding box rectangles and corresponding
    # confidence scores
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    # loop over the number of rows
    for y in range(0, numRows):
        # extract the scores (probabilities), followed by the
        # geometrical data used to derive potential bounding box
        # coordinates that surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        # loop over the number of columns
        for x in range(0, numCols):
            # if our score does not have sufficient probability,
            # ignore it
            if scoresData[x] < min_confidence:
                continue

            # compute the offset factor as our resulting feature
            # maps will be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            # extract the rotation angle for the prediction and
            # then compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # use the geometry volume to derive the width and height
            # of the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            # compute both the starting and ending (x, y)-coordinates
            # for the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            # add the bounding box coordinates and probability score
            # to our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    # return a tuple of the bounding boxes and associated confidences
    return (rects, confidences)


def detect_text(image, padding_x: float, padding_y: float):
    orig = image.copy()
    image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
    (origH, origW) = image.shape[:2]

    # set the new width and height and then determine the ratio in change
    # for both the width and height
    (newW, newH) = (east_width, east_height)
    rW = origW / float(newW)
    rH = origH / float(newH)

    # resize the image and grab the new image dimensions
    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]

    # define the two output layer names for the EAST detector model that
    # we are interested -- the first is the output probabilities and the
    # second can be used to derive the bounding box coordinates of text
    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]

    # load the pre-trained EAST text detector
    print("[INFO] loading EAST text detector...")
    net = cv2.dnn.readNet(east_loc)

    # construct a blob from the image and then perform a forward pass of
    # the model to obtain the two output layer sets
    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)

    # decode the predictions, then  apply non-maxima suppression to
    # suppress weak, overlapping bounding boxes
    (rects, confidences) = decode_predictions(scores, geometry)
    boxes = non_max_suppression(np.array(rects), probs=confidences)

    # initialize the list of results
    results = []
    # loop over the bounding boxes
    for (startX, startY, endX, endY) in boxes:
        # scale the bounding box coordinates based on the respective
        # ratios
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)

        # in order to obtain a better OCR of the text we can potentially
        # apply a bit of padding surrounding the bounding box -- here we
        # are computing the deltas in both the x and y directions
        dX = int((endX - startX) * padding_x)
        dY = int((endY - startY) * padding_y)

        # apply padding to each side of the bounding box, respectively
        startX = max(0, startX - dX)
        startY = max(0, startY - dY)
        endX = min(origW, endX + (dX * 2))
        endY = min(origH, endY + (dY * 2))

        text = "" #pytesseract.image_to_string(roi, config=config)

        # add the bounding box coordinates and OCR'd text to the list
        # of results
        results.append(((startX, startY, endX, endY), text))

    # sort the results bounding box coordinates from top to bottom
    results = sorted(results, key=lambda r:r[0][1])

    # check for collisions between the boxes
    new_results = []
    for ((a_start_x, a_start_y, a_end_x, a_end_y), _) in results:
        has_collision = False
        idx = 0
        for (b_start_x, b_start_y, b_end_x, b_end_y) in new_results:
            if check_bounding_collision((a_start_x, a_start_y, a_end_x, a_end_y), (b_start_x, b_start_y, b_end_x, b_end_y)):
                has_collision = True
                break
            else:
                idx += 1

        if has_collision:
            # determine new max/min
            min_x = 0
            min_y = 0
            max_x = 0
            max_y = 0

            if a_start_x <= b_start_x:
                min_x = a_start_x
            else:
                min_x = b_start_x
            
            if a_start_y <= b_start_y:
                min_y = a_start_y
            else:
                min_y = b_start_y

            if a_end_x >= b_end_x:
                max_x = a_end_x
            else:
                max_x = b_end_x
            
            if a_end_y >= b_end_y:
                max_y = a_end_y
            else:
                max_y = b_end_y

            # set the new bounding box
            new_results[idx] = (min_x, min_y, max_x, max_y)
            # new_results.append((a_start_x, a_start_y, a_end_x, a_end_y))
        else:
            new_results.append((a_start_x, a_start_y, a_end_x, a_end_y))

        print(str(idx))
        idx += 1


    # loop over the results
    for (startX, startY, endX, endY) in new_results:
        # strip out non-ASCII text so we can draw the text on the image
        # using OpenCV, then draw the text and a bounding box surrounding
        # the text region of the input image
        output = orig.copy()
        cv2.rectangle(output, (startX, startY), (endX, endY), (100, 200, 255), 2)
        # cv2.putText(output, text, (startX, startY - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        # show the output image
        cv2.imshow("Text Detection", output)
        cv2.waitKey(0)

"""
END EAST TEXT DETECTION
"""

"""
OLD IMPLEMENTATION
"""
def old_implementation(image_orig, image):
        # _, mask = cv2.threshold(image,180,255,cv2.THRESH_TOZERO_INV)
    # image_final = cv2.bitwise_and(image, image, mask=mask)

    # cv2.imshow("Original image", image_orig)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # _, mask = cv2.threshold(image,180,255,cv2.THRESH_TOZERO_INV)
    # _, new_img = cv2.threshold(image,150,255,cv2.THRESH_OTSU)  # for black text , cv.THRESH_BINARY_INV
    th, threshed = cv2.threshold(image, 100, 255, 
       cv2.THRESH_BINARY|cv2.THRESH_OTSU) 
    new_img = cv2.adaptiveThreshold(threshed,255,cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,11,2)

    # cv2.imshow("New image", new_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5,
                                                         3))  # to manipulate the orientation of dilution , large x means horizonatally dilating  more, large y means vertically dilating more
    dilated = cv2.dilate(new_img, kernel, iterations=3)  # dilate , more the iteration more the dilation

    cv2.imshow("Dilation", dilated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # image, contours, hierarchy = cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    contours, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # for contour in contours:
    #     [x, y, w, h] = cv2.boundingRect(contour)
    #     # get rectangle bounding contour
    #     # (x, y, w, h) = cv2.boundingRect(contour)

    #     # Don't plot small false positives that aren't text
    #     if w < 35 and h < 35:
    #         continue

    #     # draw rectangle around contour on original image
    #     img_cpy = iso_image.copy()
    #     cv2.rectangle(img_cpy, (x, y), (x+w, y+h), (150, 0, 150), 2)
    #     show(img_cpy, "Test")
    
    # END ORIGINAL

    
    # cv2.imshow('captcha_result', img_cpy)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


"""
END OLD IMPLEMENTATION
"""

"""
Isolates text on the the provide image. Blackens the 
background and whitens text.
"""
def isolate_text(image):
    _, threshed = cv2.threshold(image, 100, 255, 
       cv2.THRESH_BINARY|cv2.THRESH_OTSU) 
    new_img = cv2.adaptiveThreshold(threshed,255,cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,11,2)

    return new_img

def dilate_text(image, kernel_width = 3, kernel_height = 3, iterations = 3):
     # to manipulate the orientation of dilution , large x means horizonatally dilating  more, large y means vertically dilating more
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_width,
                                                         kernel_height)) 
    
    # dilate , more the iteration more the dilation
    dilated = cv2.dilate(image, kernel, iterations = iterations)
    return dilated

def find_contours(image):
    contours, _ = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        [x, y, w, h] = cv2.boundingRect(contour)

        # Don't plot small false positives that aren't text
        if w < 35 and h < 35:
            continue

        # draw rectangle around contour on original image
        img_cpy = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)
        cv2.rectangle(img_cpy, (x, y), (x+w, y+h), (150, 0, 150), 2)
        show(img_cpy, "Test")

"""
For the following processing functions:
@param image a cropped image that requires special processing
@return a list of tuples with the format (FieldData, image)

process_upper_image:
- Assumption: important information is located in the top right

process_middle_region:


process_lower_region:

"""
def show(img, title):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_upper_region(image):
    height = image.shape[0]
    width  = image.shape[1]

    dim_w = int(width / 2)

    left_x = 0
    right_x = left_x + dim_w

    cropped_right = image[0 : height, right_x : right_x + dim_w]
    cropped_right = isolate_text(cropped_right)

    # show(cropped_left, "Upper Left")
    # show(cropped_right, "Upper Right")

    date = FieldData()
    date.bounds = BoundingRect(right_x, 0, dim_w, height)
    date.field_type = FieldType.FIELD_TYPE_DATE

    return [(date, cropped_right)]

    # detect_text(cropped_right, .5, .1)


def process_middle_region(image):
    height = image.shape[0]
    width  = image.shape[1]

    dim_h = int(height / 3)
    removed_upper_left = 150 # number of pixels to remove from this region

    top_y = 0
    lower_y = top_y + dim_h - 20

    # UPPER REGION
    # -------------------------------------
    cropped_upper = image[top_y : top_y + dim_h, removed_upper_left : width]
    left_upper = cropped_upper[0: cropped_upper.shape[0], 0 : int(cropped_upper.shape[1] / 3) * 2]
    right_upper = cropped_upper[0: cropped_upper.shape[0], int(cropped_upper.shape[1] / 3) * 2 : cropped_upper.shape[1]]
    
    # Pay to the order of field
    iso_left_upper = isolate_text(left_upper)
    # show(iso_left_upper, "Pay to the order of")

    pay = FieldData()
    pay.bounds = BoundingRect(removed_upper_left, 0, int(cropped_upper.shape[1] / 3) * 2, dim_h)
    pay.field_type = FieldType.FIELD_TYPE_PAY_TO_ORDER_OF
    
    # amount field
    iso_right_upper = isolate_text(right_upper)
    # show(iso_right_upper, "Amount")

    amount = FieldData()
    amount.bounds = BoundingRect(int(cropped_upper.shape[1] / 3) * 2, 
                                 0, 
                                 cropped_upper.shape[1] - int(cropped_upper.shape[1] / 3) * 2, 
                                 height)
    amount.field_type = FieldType.FIELD_TYPE_AMOUNT

    # LOWER REGION
    #-------------------------------------
    cropped_lower = image[lower_y : height, 0 : width]
    lower_left = cropped_lower[0 : cropped_lower.shape[0], 0 : int(cropped_lower.shape[1] / 4) * 3]

    # Written amount
    iso_lower = isolate_text(lower_left)    
    # show(iso_lower, "Written Amount")

    written_amount = FieldData()
    written_amount.bounds = BoundingRect(0, lower_y, int(cropped_lower.shape[1] / 4) * 3, height - lower_y)
    written_amount.field_type = FieldType.FIELD_TYPE_AMOUNT_WRITTEN

    return [(pay, iso_left_upper), 
            (amount, iso_right_upper), 
            (written_amount, iso_lower)
           ]

    # detect_text(iso_upper, .8, .6)
    # detect_text(iso_lower, .8, .4)

"""
NOTE(Dustin): Idea: find the middle two lines. Which one is longer? Found the written amount. 
Above it will be the "Pay to the order of" field and amount. 

Locatlity of pixel colors. For each pixel, check its neighbors. 
If pixels to the left or right are black, potentially found a line. Provide a 
threshold for a length of a line. After a line scan, write the pixels to a 
copy image. 

"""
# import numpy as np
# from matplotlib import pyplot as plt
def detect_lines(img, color):

    kernel_size = 5
    blur_gray = cv2.GaussianBlur(img,(kernel_size, kernel_size),0)
    edges = cv2.Canny(blur_gray,50,150)

    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 100  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 100  # minimum number of pixels making up a line
    max_line_gap = 5  # maximum gap in pixels between connectable line segments
    blank = img.copy() * 0

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                   min_line_length, max_line_gap)

    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(blank,(x1,y1),(x2,y2),color,5)

    cv2.imshow("Lines", blank)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_lower_region(image):
    height = image.shape[0]
    width  = image.shape[1]

    iso_image = isolate_text(image)
    dilated = dilate_text(iso_image, kernel_width = 5, kernel_height = 3, iterations = 3)
    # find_contours(dilated)

    # show(iso_image, "Lower region")

    lower = FieldData()
    lower.bounds = BoundingRect(0, 0, width, height)
    lower.field_type = FieldType.FIELD_TYPE_NONE

    return [(lower, iso_image)]

    # detect_lines(iso_image, (255,255,255))
    # detect_text(iso_image, .1, .1)
    

"""
Provided an image of a check, a series of fields are
extracted from the image. For each field found, a FieldData
struct is created. This struct contains default values except
for the bounding box. The bounding box of the field is recorded
for further processing in a later pipelin stage.

@param image: Image of the check to extract fields from

@return A list of tuples containing a DataPair and a cropped image.
"""
def extractFieldsEntryPoint(image_orig, image):

    # Split the image into thirds
    height = image.shape[0]
    width  = image.shape[1]

    dim_w = int(width / 3)
    dim_h = int(height / 3)
    
    upper_x = 0
    upper_y = 0

    middle_x = 0
    middle_y = upper_y + dim_h

    lower_x = 0
    lower_y = middle_y + dim_h

    # draw the bounding region for visualization
    img_cpy = image.copy()
    cv2.rectangle(img_cpy, (upper_x, upper_y),   (upper_x + width, upper_y + dim_h),   (128, 0, 128), 2)
    cv2.rectangle(img_cpy, (middle_x, middle_y), (middle_x + width, middle_y + dim_h), (128, 0, 128), 2)
    cv2.rectangle(img_cpy, (lower_x, lower_y),   (lower_x + width, lower_y + dim_h),   (128, 0, 128), 2)

    # crop each section
    cropped_upper  = img_cpy[upper_y  : upper_y  + dim_h, upper_x  : width]
    cropped_middle = img_cpy[middle_y : middle_y + dim_h, middle_x : width]
    cropped_lower  = img_cpy[lower_y  : lower_y  + dim_h, lower_x  : width]

    # process each section
    upper_images  = process_upper_region(cropped_upper)
    middle_images = process_middle_region(cropped_middle)
    lower_images  = process_lower_region(cropped_lower)

    # compile the list of fields
    list = []
    for img in upper_images:
        list.append(img)
    
    for img in middle_images:
        list.append(img)
    
    for img in lower_images:
        list.append(img)

    return img_cpy, list
