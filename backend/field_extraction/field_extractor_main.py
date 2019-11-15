"""
Attempted approaches:
- Naive aabb detection
  - the problem with approach is the spacing between characters and lines are inconsistent

- Splitting the image into pieces
  - checks have an inconsistent format for spacing. makes it difficult to, say, pull apart the middle region

- EAST text detection
  - handwriting has inconsentent spacing, which throws the algorithm off. Bounding boxes overlap and are
    inconsistent, and needed a aabb merger. Even then, there are inconsistent results


TODO(Dustin): Make sure during field extraction, all process call process_field
rather than running their own function
"""


import backend.data_extraction.field.data.field_data           as field_data
import backend.utils.cv_utils                                  as cv_utils

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
    # print("Checking collision...")
    # print("    Box A: " + str(boxA))
    # print("    Box B: " + str(boxB)) 

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
    cv_utils.show(image, "Detecting text image")
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
    # for (startX, startY, endX, endY) in new_results:
    for ((startX, startY, endX, endY), _) in results:
        # strip out non-ASCII text so we can draw the text on the image
        # using OpenCV, then draw the text and a bounding box surrounding
        # the text region of the input image
        output = orig.copy()
        cv2.rectangle(output, (startX, startY), (endX, endY), (100, 200, 255), 2)
        # cv2.putText(output, text, (startX, startY - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        # show the output image
        cv_utils.show(output, "Text Detection")

"""
END EAST TEXT DETECTION
"""

"""
Isolates text on the the provide image. Blackens the 
background and whitens text.
"""
def isolate_text(image):
    _, threshed = cv2.threshold(image, 100, 255, 
       cv2.THRESH_OTSU) 
    # cv2.THRESH_BINARY|
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
        cv_utils.show(img_cpy, "Test")

"""
Merge bounding boxes that are close together based on an
x, y threshold.
"""
def merge_close_bb(image, bb_list, x_threshold = 0, y_threshold = 0):
    new_list = []

    for (a_start_x, a_start_y, a_width, a_height) in bb_list:
        can_merge = False
        idx = 0

        a_end_x = a_start_x + a_width
        a_end_y = a_start_y + a_height

        for (b_start_x, b_start_y, b_width, b_height) in new_list:
            
            b_end_x = b_start_x + b_width
            b_end_y = b_start_y + b_height

            # adjust second box based on the threshold 
            b_new_start_x = b_start_x - x_threshold
            b_new_start_y = b_start_y - y_threshold

            b_new_end_x = b_end_x + x_threshold
            b_new_end_y = b_end_y + y_threshold

            # see if the boxes overlap
            if check_bounding_collision((a_start_x, a_start_y, a_end_x, a_end_y), 
                                        (b_new_start_x, b_new_start_y, b_new_end_x, b_new_end_y)):
                can_merge = True

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

                # update the box
                new_list[idx] = (min_x, min_y, max_x - min_x, max_y - min_y)

                break

            idx += 1
        # end loop

        if not can_merge:
            new_list.append((a_start_x, a_start_y, a_end_x - a_start_x, a_end_y - a_start_y))
    # end loop

    return new_list

"""
Merge overlapping bounding boxes and return a new list.

@param bb_list should be a list of tuples (x, y, w, h)

@return a list of tuples (x, y, w, h)
"""
def merge_overlapping_bb(image, bb_list):
    new_list = []
    # check_bounding_collision(boxA: tuple, boxB: tuple)
    # box = (start_x, start_y, end_x, end_yu)
    for ((a_start_x, a_start_y, a_width, a_height)) in bb_list:
        has_collision = False
        idx = 0

        a_end_x = a_start_x + a_width
        a_end_y = a_start_y + a_height

        for (b_start_x, b_start_y, b_width, b_height) in new_list:

            b_end_x = b_start_x + b_width
            b_end_y = b_start_y + b_height

            if check_bounding_collision((a_start_x, a_start_y, a_end_x, a_end_y), (b_start_x, b_start_y, b_end_x, b_end_y)):
                has_collision = True
                break
            else:
                idx += 1

        # end loop 

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
            new_list[idx] = (min_x, min_y, max_x - min_x, max_y - min_y)
            # new_list.append((a_start_x, a_start_y, a_end_x - a_start_x, a_end_y - a_start_y))
        else:
            new_list.append((a_start_x, a_start_y, a_end_x - a_start_x, a_end_y - a_start_y))

    # end loop

    return new_list

def draw_rects(image, rects):
    img_cpy = image.copy()
    for (x, y, w, h) in rects:
        cv2.rectangle(img_cpy, (x, y), (x+w, y+h), (150, 0, 150), 2)
    cv_utils.show(img_cpy, "Drawing rectangles")



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

    white = (255, 255, 255)
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 100  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 100  # minimum number of pixels making up a line
    max_line_gap = 50  # maximum gap in pixels between connectable line segments

    # sets image to black
    blank = img.copy() * 0
    # turns it white
    # blank = np.full(blank.shape, 255, np.uint8)

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                   min_line_length, max_line_gap)

    if lines is None:
        return

    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(blank,(x1,y1),(x2,y2),white,1)

    final = cv2.bitwise_or(img, blank)
    cv_utils.show(final, "Lines")
    return final


"""
Process the date field type.

@return a list of bounding boxes contained in the 
tuple (x_start, y_start, width, height)
"""
def process_date(image):
    height = image.shape[0]
    width  = image.shape[1]

    regions = cv_utils.impl_mser(image)

    # Get all possible bounding regions
    img_cpy = image.copy()
    list = []
    for box in regions:
        [x, y, w, h] = cv2.boundingRect(box)

        if w < 35 and h < 35:
            continue

        if h < 5:
            continue

        # if the bb is to tall, then it is probably incorrect
        if h >= int(height / 2):
            continue

        # cv2.rectangle(img_cpy, (x, y), (x+w, y+h), (150, 0, 150), 2)
        list.append((x, y, w, h))
    
    # draw_rects(image, list)

    # Merge overlapping regions
    merged_list = merge_overlapping_bb(image, list)
    
    # draw_rects(image, merged_list)
    
    merged_list = merge_close_bb(image, merged_list, x_threshold=30)

    # draw_rects(image, merged_list)

    return merged_list

"""
TODO(Dustin): If the generated bb is above a specified size, 
remove it. So like, 90% of the image
"""
def process_field(image, expand_x = 0, expand_y = 0, threshold_x = 0, threshold_y = 0):
    height = image.shape[0]
    width  = image.shape[1]

    regions = cv_utils.impl_mser(image)

    list = []
    for box in regions:
        [x, y, w, h] = cv2.boundingRect(box)

        if w < 35 and h < 35:
            continue

        if h < 5:
            continue

        if w >= int(width * .80) and h >= int(height * .80):
            continue

        # cv2.rectangle(img_cpy, (x, y), (x+w, y+h), (150, 0, 150), 2)
        e_width  = min(width, w+expand_x)
        e_height = min(height, h+expand_y)
        list.append((x, y, e_width, e_height))
    # end for

    # draw_rects(image, list)

    # Merge overlapping regions
    merged_list = merge_overlapping_bb(image, list)
    
    # draw_rects(image, merged_list)
    
    merged_list = merge_close_bb(image, merged_list, threshold_x, threshold_y)

    # draw_rects(image, merged_list)

    return merged_list
# end function

"""
Process the amount field with the given image.

@return a list of bounding boxes (x start pixel, y start pixel, width, height)
"""
def process_amount(image):
    height = image.shape[0]
    width  = image.shape[1]

    regions = cv_utils.impl_mser(image)

    list = []
    for box in regions:
        [x, y, w, h] = cv2.boundingRect(box)

        if w < 35 and h < 35:
            continue

        if h < 5:
            continue

        # cv2.rectangle(img_cpy, (x, y), (x+w, y+h), (150, 0, 150), 2)
        e_width  = min(width, w+10)
        e_height = min(height, h+10)
        list.append((x, y, e_width, e_height))
    # end for

    # draw_rects(image, list)

    # Merge overlapping regions
    merged_list = merge_overlapping_bb(image, list)
    
    # draw_rects(image, merged_list)
    
    merged_list = merge_close_bb(image, merged_list, x_threshold=5)

    # draw_rects(image, merged_list)

    return merged_list
# end function

"""
For the following processing functions:
@param image a cropped image that requires special processing
@return a list of tuples with the format (FieldData, image)

process_upper_image:
- Assumption: important information is located in the top right

process_middle_region:


process_lower_region:
"""
def process_upper_region(image):
    height = image.shape[0]
    width  = image.shape[1]

    dim_w = int(width / 2)

    left_x = 0
    right_x = left_x + dim_w

    cropped_right = image[0 : height, right_x : right_x + dim_w]
    bb_list = process_date(cropped_right)

    possible_dates = []
    for (x, y, w, h) in bb_list:
        img_cpy = cropped_right.copy()

        img_cpy = img_cpy[y : y + h, x : x + w]
        img_cpy = isolate_text(img_cpy)
        img_cpy = cv2.medianBlur(img_cpy, 3)

        date = FieldData()
        date.bounds = BoundingRect(x, y, w, h)
        date.field_type = FieldType.FIELD_TYPE_DATE

        possible_dates.append((date, img_cpy))
    # end loop

    return possible_dates
# end function

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
    # iso_left_upper = isolate_text(left_upper)
    # show(iso_left_upper, "Pay to the order of")

    pay_order_bb = process_field(left_upper, 10, 10, 50)
    possible_pay_order = []
    for (x, y, w, h) in pay_order_bb:
        # prep image
        img_cpy = left_upper.copy()
        img_cpy = img_cpy[y : y + h, x : x + w]
        # img_cpy = isolate_text(img_cpy)
        img_cpy = cv2.medianBlur(img_cpy, 3)

        pay_order = FieldData()
        pay_order.bounds = BoundingRect(x, y, w, h)
        pay_order.field_type = FieldType.FIELD_TYPE_PAY_TO_ORDER_OF

        possible_pay_order.append((pay_order, img_cpy))
    # end for

    idx = 0
    for (pay, img) in possible_pay_order:
        img_cpy = img.copy()
        cv_utils.show(img_cpy, "IMAGE")

        detect_lines(img_cpy, (128, 128, 0))


    # amount field
    amount_bb = process_field(right_upper, 10, 10, 50)
    possible_amount = []
    for (x, y, w, h) in amount_bb:
        # prep image
        img_cpy = right_upper.copy()
        img_cpy = img_cpy[y : y + h, x : x + w]
        # img_cpy = isolate_text(img_cpy)
        img_cpy = cv2.medianBlur(img_cpy, 3)

        amount = FieldData()
        amount.bounds = BoundingRect(x, y, w, h)
        amount.field_type = FieldType.FIELD_TYPE_AMOUNT

        possible_amount.append((amount, img_cpy))
    # end for

    # idx = 0
    # for (amount, img) in possible_amount:
    #     # rect = amount.bounds
    #     img_cpy = img.copy()
    #     cv_utils.show(img_cpy, "IMAGE")
    #     detect_lines(img_cpy, (128, 128, 0))
        
    #     msers = cv_utils.impl_mser(img)

    #     for box in msers:
    #         img_cpy = img.copy()
    #         height = img_cpy.shape[0]
    #         width  = img_cpy.shape[1]

    #         (x, y, w, h) = cv2.boundingRect(box)

    #         if w >= int(width * .80) and h >= int(height * .80):
    #             continue

    #         cv2.rectangle(img_cpy, (x, y), (x + w, y + h), (128, 0, 128), 2)
    #         cv_utils.show(img_cpy, "Another mser " + str(idx))
    #     # end for
    #     idx += 1
    # end for

    # LOWER REGION
    #-------------------------------------
    cropped_lower = image[lower_y : height, 0 : width]
    lower_left = cropped_lower[0 : cropped_lower.shape[0], 0 : int(cropped_lower.shape[1] / 4) * 3]
    
    written_amount_bb = process_field(lower_left, 100, 10, 500)
    possible_written_amount = []
    for (x, y, w, h) in written_amount_bb:
        # prep image
        img_cpy = lower_left.copy()
        img_cpy = img_cpy[y : y + h, x : x + w]
        # img_cpy = isolate_text(img_cpy)
        img_cpy = cv2.medianBlur(img_cpy, 3)

        written = FieldData()
        written.bounds = BoundingRect(x, y, w, h)
        written.field_type = FieldType.FIELD_TYPE_AMOUNT_WRITTEN

        possible_written_amount.append((written, img_cpy))
    # end for

    """
    # TEST FOR FINDING ALL BOXES IN SINGLE IMAGE
    # -------------------------------------------------------------
    total = process_field(image, 100, 0, 10, 100)
    possible_total = []
    for (x, y, w, h) in total:
        # prep image
        img_cpy = image.copy()
        img_cpy = img_cpy[y : y + h, x : x + w]
        # img_cpy = isolate_text(img_cpy)
        img_cpy = cv2.medianBlur(img_cpy, 3)

        written = FieldData()
        written.bounds = BoundingRect(x, y, w, h)
        written.field_type = FieldType.FIELD_TYPE_AMOUNT_WRITTEN

        cv_utils.show(img_cpy, "Total box")

        possible_total.append((written, img_cpy))
    # end for
    return []
    """
    return possible_pay_order + possible_amount + possible_written_amount
# end function

def process_lower_region(image):
    height = image.shape[0]
    width  = image.shape[1]

    iso_image = isolate_text(image)
    dilated = dilate_text(iso_image, kernel_width = 5, kernel_height = 3, iterations = 3)
    # find_contours(dilated)

    # show(iso_image, "Lower region")

    # Routing/Account Image is just the image
    routing = FieldData()
    routing.bounds = BoundingRect(0, 0, width, height)
    routing.field_type = FieldType.FIELD_TYPE_ROUTING
    possible_routing = [(routing, image)]

    # Memo
    left_img = image[0 : int(height / 2), 0 : int(width / 2)]
    memo_bb = process_field(left_img, 10, 10, 50)
    possible_memo = []
    for (x, y, w, h) in memo_bb:
        # prep image
        img_cpy = left_img.copy()
        img_cpy = img_cpy[y : y + h, x : x + w]
        # img_cpy = isolate_text(img_cpy)
        img_cpy = cv2.medianBlur(img_cpy, 3)

        pay_order = FieldData()
        pay_order.bounds = BoundingRect(x, y, w, h)
        pay_order.field_type = FieldType.FIELD_TYPE_MEMO

        possible_memo.append((pay_order, img_cpy))
    # end for


    # Signature
    right_img = image[0 : int(height / 2), int(width / 2) : width]
    sig_bb = process_field(right_img, 10, 10, 50)
    possible_sig = []
    for (x, y, w, h) in sig_bb:
        # prep image
        img_cpy = right_img.copy()
        img_cpy = img_cpy[y : y + h, x : x + w]
        # img_cpy = isolate_text(img_cpy)
        img_cpy = cv2.medianBlur(img_cpy, 3)

        signature = FieldData()
        signature.bounds = BoundingRect(x, y, w, h)
        signature.field_type = FieldType.FIELD_TYPE_SIGNATURE

        possible_sig.append((signature, img_cpy))
    # end for

    return possible_routing + possible_memo + possible_sig
# end function

"""
To split an image, need: img, x_start, y_start,
x_max, y_max

@param img: image to split
@param min_x
@param min_y
@param max_x
@param max_y

@return new_image the region to crop
@return old_image the old image

Format of return:
(new_image, old_image)
"""
def crop(img, min_x, min_y, max_x, max_y):
    new_image = img[min_y : max_y,  min_x : max_x]
    return (new_image, img)

def upper_extract(image):
    height = image.shape[0]
    width  = image.shape[1]

    # crop right half
    min_x = int(width * 0.5)
    max_x = width
    min_y = int(height * 0.5)
    max_y = height

    date_img = image[min_y  : max_y,  min_x : max_x]

    date = FieldData()
    date.bounds = BoundingRect(min_x, min_y, max_x - min_x, max_y - min_y)
    date.field_type = FieldType.FIELD_TYPE_DATE

    return [(date, date_img)]

def middle_extract(image):
    height = image.shape[0]
    width  = image.shape[1]

    # Get the pay row
    pay_min_x = 0
    pay_max_x = width
    pay_min_y = 0
    pay_max_y = int(height * 0.40)
    pay_width  = pay_max_x - pay_min_x
    pay_height = pay_max_y - pay_min_y
    prow_img = image[pay_min_y : pay_max_y, pay_min_x : pay_max_x]

    # Get the pay to the order of field bounds
    pfield_min_x = int(pay_width * 0.12)
    pfield_max_x = int(pay_width * 0.73)
    pfield_min_y = int(pay_height * 0.0)
    pfield_max_y = int(pay_height * 1.0)
    pfield_width  = pfield_max_x - pfield_min_x
    pfield_height = pfield_max_y - pfield_min_y

    # Get the amount bounds
    amount_min_x = int(pay_width * 0.775)
    amount_max_x = int(pay_width * 0.950)
    amount_min_y = 0
    amount_max_y = pay_height
    amount_width  = amount_max_x - amount_min_x
    amount_height = amount_max_y - amount_min_y

    # Get the written row
    wrow_min_x = 0
    wrow_max_x = width
    wrow_min_y = int(height * 0.40)
    wrow_max_y = int(height * 1.00)
    wwidth  = wrow_max_x - wrow_min_x
    wheight = wrow_max_y - wrow_min_y
    wrow_img = image[wrow_min_y : wrow_max_y, wrow_min_x : wrow_max_x]

    # Get the written amount field bounds
    written_min_x = int(wwidth * 0.05)
    written_max_x = int(wwidth * 0.75)
    written_min_y = 0
    written_max_y = wheight
    written_width  = written_max_x - written_min_x
    written_height = written_max_y - written_min_y

    # Create the pay field
    pay_img = prow_img[pfield_min_y : pfield_max_y,  pfield_min_x : pfield_max_x]
    pay = FieldData()
    pay.bounds = BoundingRect(pfield_min_x, pfield_min_y, pfield_width, pfield_height)
    pay.field_type = FieldType.FIELD_TYPE_PAY_TO_ORDER_OF
    pay_field = (pay, pay_img)

    # Create the amount field
    amount_img = prow_img[amount_min_y : amount_max_y,  amount_min_x : amount_max_x]
    amount = FieldData()
    amount.bounds = BoundingRect(amount_min_x, amount_min_y, amount_width, amount_height)
    amount.field_type = FieldType.FIELD_TYPE_AMOUNT
    amount_field = (amount, amount_img)

    # Create the pay field
    written_img = wrow_img[written_min_y : written_max_y,  written_min_x : written_max_x]
    written = FieldData()
    written.bounds = BoundingRect(written_min_x, written_min_y, written_width, written_height)
    written.field_type = FieldType.FIELD_TYPE_AMOUNT_WRITTEN
    written_field = (written, written_img)

    return [pay_field, amount_field, written_field]

def lower_extract(image):
    height = image.shape[0]
    width  = image.shape[1]

    # Get the top row
    top_min_x = 0
    top_max_x = width
    top_min_y = int(height * 0.05)
    top_max_y = int(height * 0.65)
    top_width  = top_max_x - top_min_x
    top_height = top_max_y - top_min_y
    top_img = image[top_min_y : top_max_y, top_min_x : top_max_x]

    # Get the memo region bounds
    memo_min_x = int(top_width * 0.08)
    memo_max_x = int(top_width * 0.45)
    memo_min_y = 0
    memo_max_y = top_height
    memo_width  = memo_max_x - memo_min_x
    memo_height = memo_max_y - memo_min_y

    # Get the signature region bounds
    sig_min_x = int(top_width * 0.50)
    sig_max_x = int(top_width * 0.92)
    sig_min_y = 0
    sig_max_y = top_height
    sig_width  = sig_max_x - sig_min_x
    sig_height = sig_max_y - sig_min_y

    # Get the bottom row
    acc_min_x = 0
    acc_max_x = width
    acc_min_y = int(height * 0.55)
    acc_max_y = int(height * 1.00)
    acc_width  = acc_max_x - acc_min_x
    acc_height = acc_max_y - acc_min_y
    low_img = image[acc_min_y : acc_max_y, acc_min_x : acc_max_x]

    # Create the memo field
    memo_img = top_img[memo_min_y : memo_max_y,  memo_min_x : memo_max_x]
    memo = FieldData()
    memo.bounds = BoundingRect(memo_min_x, memo_min_y, memo_width, memo_height)
    memo.field_type = FieldType.FIELD_TYPE_MEMO
    memo_field = (memo, memo_img)

    # Create the signature field
    sig_img = top_img[sig_min_y : sig_max_y,  sig_min_x : sig_max_x]
    sig = FieldData()
    sig.bounds = BoundingRect(sig_min_x, sig_min_y, sig_width, sig_height)
    sig.field_type = FieldType.FIELD_TYPE_SIGNATURE
    sig_field = (sig, sig_img)

    # Create the routing/account field
    rout_img = image[acc_min_y : acc_max_y,  acc_min_x : acc_max_x]
    rout = FieldData()
    rout.bounds = BoundingRect(acc_min_x, acc_min_y, acc_width, acc_height)
    rout.field_type = FieldType.FIELD_TYPE_ROUTING
    rout_field = (rout, rout_img)

    acc = FieldData()
    acc.bounds = BoundingRect(acc_min_x, acc_min_y, acc_width, acc_height)
    acc.field_type = FieldType.FIELD_TYPE_ACCOUNT
    acc_field = (acc, rout_img)

    return [memo_field, sig_field, rout_field, acc_field]

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

    dim_middle_h = int(height / 4)

    upper_x_start = 0
    upper_x_end   = width
    upper_y_start = 0
    upper_y_end   = int(height * .33)

    middle_x_start = 0
    middle_x_end   = width
    middle_y_start = int(height * .30)
    middle_y_end   = int(height * .70)

    lower_x_start = 0
    lower_x_end   = width
    lower_y_start = int(height * .70)
    lower_y_end   = int(height * .90)

    # draw the bounding region for visualization
    img_cpy = image.copy()
    # cv2.rectangle(img_cpy, (upper_x, upper_y),   (upper_x + width, upper_y + dim_h),   (128, 0, 128), 2)
    # cv2.rectangle(img_cpy, (middle_x, middle_y), (middle_x + width, middle_y + dim_h), (128, 0, 128), 2)
    # cv2.rectangle(img_cpy, (lower_x, lower_y),   (lower_x + width, lower_y + dim_h),   (128, 0, 128), 2)

    # crop each section
    cropped_upper  = img_cpy[upper_y_start  : upper_y_end,  upper_x_start : upper_x_end]
    cropped_middle = img_cpy[middle_y_start : middle_y_end, upper_x_start : upper_x_end]
    cropped_lower  = img_cpy[lower_y_start  : lower_y_end,  upper_x_start : upper_x_end]

    # process each section
    #upper_images  = process_upper_region(cropped_upper) # BB WORKING
    #middle_images = process_middle_region(cropped_middle)
    #lower_images  = process_lower_region(cropped_lower)

    upper_images  = upper_extract(cropped_upper) # BB WORKING
    middle_images = middle_extract(cropped_middle)
    lower_images  = lower_extract(cropped_lower)

    # compile the list of fields
    list = []
    for img in upper_images:
        list.append(img)
    
    for img in middle_images:
        list.append(img)
    
    for img in lower_images:
        list.append(img)

    return img_cpy, list
