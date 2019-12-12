"""
NOTE(Dustin): Within this module there are several approaches to extraction fields from a check image.
While many of these strategies are not currently in use, we feel that they can be utilized
under specific circumstances. This following is a destription of the strategies that have
been implemented, and how they might be used.

-----------------------------------------------------------------------------------------------------------
- EAST text detection
-----------------------------------------------------------------------------------------------------------
EAST edge detection is a deep learning model that is pretrained and provided by OpenCV. This algorithm
requires a single companion file located in resources/east/frozen_east_text_detection.pb.

It does a good job at recognizing text on an image, but it struggles at finding accurate bounding
boxes over the text. In order to mitigate this weakness, we implemented the following hueristics:
  - Merge Bounding Boxes: many of the bounding boxes found by the algorithm were small and
    disjointed, so all overlapping boxes are merged.
  - Merge Nearby Bounding Boxes: merging overlapping bounding boxes was not enough as handwritten
    text can be abnormally spaced, so an alogirithm for merging nearby bounding boxes was implemented.
    The caller can choose to provide a threshold where all boxes overlapping with the threshold
    applied will be merged.

Even with these hueristics in place, this algorithm was inconsistent due to the randomness
of human handwriting. Furthermore, the data extraction required that no machine text be present
in a cropped image, but this approach could not guarentee this. 

-----------------------------------------------------------------------------------------------------------
- Computer Vision Techniques
-----------------------------------------------------------------------------------------------------------
OpenCV provides a lot of computer vision techniques that do not use machine learning, which we thought 
could decrease runtime processing. There are several approaches located in this module that have internal
parameters that can be tweaked to a developers liking. As with the EAST Edge Detection, these approaches
tended to fail due to the randomness with human handwritting. However, we believe they can work on smaller,
more isolated examples.
  - Text Isolation is an approach to isolate the foreground from the background. This approach was fairly
    consistent in isolating the text from the background when paired with text dilation.
  - Text Dilation was utilized after isolating the text. The black/white color of the text is dilated, so
    that it overlaps with other character. Dilation was often paired with line contouring.
  - Line Contouring finding the bounding region around text. While unpredictible by itself, it is very useful
    if the text has been dilated. At this point, it can assumed that regions of text has been turned into
    a single "blob", making contouring straightforward. From here, bounding regions can be determined.
  - Line Detection was utilized using a Hough Lines hueristic that would detect all lines in an image. We 
    found that these lines often corrupted output during th data extraction, so an approach was needed to
    remove them. Hough Lines was only moderately successful, but ofter detected text as lines, which is not
    what we wanted.    

-----------------------------------------------------------------------------------------------------------
- Training of a Model
-----------------------------------------------------------------------------------------------------------
This final approach provided the most consistent approach to detecting fields, so field extraction currently
utilizes this approach. There are two steps: an offline and an online steps. The offline step is discussed in
test/field_extraction/test_field_extraction.py. Long story short, it finds the optimal regions for each field
given a type of check. This module takes these parameters and crops the image based on them. There are three
routines for this:
  - process_upper
  - process_lower
  - process_middle
It is assumed that:
  - date is in the upper region
  - pay to the order of, amount, and written amount is located in the middle region
  - signature, memo, routing, and account number is in the lower region.

The major weakness of this approach is that we have to have parameters to a specific type of check in order
to obtain consistent output. This approach could benefit greatly from more machine learning approach that
would allow it to handle multiple types of checks.

"""

from ..utils.cv_utils import show, impl_mser

from ..data_extraction.field.data.field_data import FieldType, FieldData, BoundingRect

import numpy as np
import cv2

from imutils.object_detection import non_max_suppression


"""
Start of the EAST Text detection algorithm.

Adapted from: https://www.pyimagesearch.com/2018/09/17/opencv-ocr-and-text-recognition-with-tesseract/

"""
east_loc = "resources/east/frozen_east_text_detection.pb"
min_confidence = 0.5 
east_width = 320  # must be multiple of 32
east_height = 320  # must be multiple of 32

"""
A box has the tuple: (startX, startY, endX, endY)

@param boxA: first bounding box
@param boxB: second bounding box 
"""
def check_bounding_collision(boxA: tuple, boxB: tuple):
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

"""
Merge bounding boxes that are close together based on an
x, y threshold.

@param image: the image to extract fields from
@param bb_list: list of bounding boxes
@param x_threshold: threshold for the x value, used for merging and combining of overlapping bounding boxes
@param y_threshold: threshold for y value, used similar to above

@return a list of tuples (x, y, w, h)
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

@param image: image whose fields are being extracted
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


"""
Determines confidence levels of the accuracy of extracted bounding boxes

@param scores: the location to store the scores
@param geometry: contains the dimensions and coordinates of bounding boxes

@return tuple: containing the bounding boxes and the confidence levels of each
"""


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
    return rects, confidences


"""
Method to detect text

@param image: image to extract field from
@param padding_x: padding for the x coordinate
@param padding_y: padding for the y coordinate 
"""


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


"""
END EAST TEXT DETECTION
"""

"""
Isolates text on the the provide image. Blackens the 
background and whitens text.

@param image: image to modify

@return new_img: the modified image
"""


def isolate_text(image):
    _, threshed = cv2.threshold(image, 100, 255, 
       cv2.THRESH_OTSU) 
    # cv2.THRESH_BINARY|
    new_img = cv2.adaptiveThreshold(threshed,255,cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,11,2)

    return new_img


"""
Dilutes the text in the image

@param image: image to dilute
@param kernel_width: width of kernel for diluting
@param kernel_height: height of kernel for diluting
@param iterations: number of iterations of dilation

@return dilated: the dilated image
"""


def dilate_text(image, kernel_width = 3, kernel_height = 3, iterations = 3):
     # to manipulate the orientation of dilution , large x means horizonatally dilating  more, large y means vertically dilating more
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_width,
                                                         kernel_height)) 
    
    # dilate , more the iteration more the dilation
    dilated = cv2.dilate(image, kernel, iterations = iterations)
    return dilated


"""
Finds the contours in the image

@param image: the image to find contours in
"""


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
Draws rectangles on the image

@param image: image to draw rectangles on
@param rects: locations/coordinates of rectangles to draw
"""


def draw_rects(image, rects):
    img_cpy = image.copy()
    for (x, y, w, h) in rects:
        cv2.rectangle(img_cpy, (x, y), (x+w, y+h), (150, 0, 150), 2)
    show(img_cpy, "Drawing rectangles")


"""
Detects lines in the image

@param img: the image to detect lines in
@param color: the color of the image

@return final: the result of the line detection
"""


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
    show(final, "Lines")
    return final


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


"""
Extracts the upper region of the image

@param image: image to extract upper region of

@return tuple containing the date FieldData object and cropped image of date
"""


def upper_extract(image):
    height = image.shape[0]
    width  = image.shape[1]

    # crop right half
    min_x = int(width * 0.65)
    max_x = int(width * 0.85)
    min_y = int(height * 0.55)
    max_y = int(height * 0.9)

    date_img = image[min_y  : max_y,  min_x : max_x]

    date = FieldData()
    date.bounds = BoundingRect(min_x, min_y, max_x - min_x, max_y - min_y)
    date.field_type = FieldType.FIELD_TYPE_DATE

    return [(date, date_img)]


"""
Extracts the middle region of the image

@param image: image to extract middle region from

@return tuple containing the Pay to the Order Of FieldData object, Amount FieldData and AmountWritten FieldData objects
"""


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
    pstart_x = pay_min_x + pfield_min_x
    pstart_y = pay_min_y + pfield_min_y

    # Get the amount bounds
    amount_min_x = int(pay_width * 0.795)
    amount_max_x = int(pay_width * 0.950)
    amount_min_y = 0
    amount_max_y = pay_height
    amount_width  = amount_max_x - amount_min_x
    amount_height = amount_max_y - amount_min_y
    amount_start_x = pay_min_x + amount_min_x
    amount_start_y = pay_min_y + amount_min_y

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
    written_max_y = int(wheight*0.5)
    written_width  = written_max_x - written_min_x
    written_height = written_max_y - written_min_y
    written_start_x = wrow_min_x + written_min_x
    written_start_y = wrow_min_y + written_min_y

    # Create the pay field
    pay_img = prow_img[pfield_min_y : pfield_max_y,  pfield_min_x : pfield_max_x]
    pay = FieldData()
    pay.bounds = BoundingRect(pstart_x, pstart_y, pfield_width, pfield_height)
    pay.field_type = FieldType.FIELD_TYPE_PAY_TO_ORDER_OF
    pay_field = (pay, pay_img)

    # Create the amount field
    amount_img = prow_img[amount_min_y : amount_max_y,  amount_min_x : amount_max_x]
    amount = FieldData()
    amount.bounds = BoundingRect(amount_start_x, amount_start_y, amount_width, amount_height)
    amount.field_type = FieldType.FIELD_TYPE_AMOUNT
    amount_field = (amount, amount_img)

    # Create the pay field
    written_img = wrow_img[written_min_y : written_max_y,  written_min_x : written_max_x]
    written = FieldData()
    written.bounds = BoundingRect(written_start_x, written_start_y, written_width, written_height)
    written.field_type = FieldType.FIELD_TYPE_AMOUNT_WRITTEN
    written_field = (written, written_img)

    return [pay_field, amount_field, written_field]


"""
Extracts the lower region of the image

@param image: the image whose lower region to extract

@return tuple containing the Memo FieldData object, Signature FieldData object, RoutingNumber FieldData object, and 
AccountNumber FieldData object
"""


def lower_extract(image):
    height = image.shape[0]
    width  = image.shape[1]

    # show(image, "lower")

    # Get the top row
    top_min_x = 0
    top_max_x = width
    top_min_y = int(height * 0.05)
    top_max_y = int(height * 0.65)
    top_width  = top_max_x - top_min_x
    top_height = top_max_y - top_min_y
    top_img = image[top_min_y : top_max_y, top_min_x : top_max_x]

    # Get the memo region bounds
    memo_min_x = int(top_width * 0.09)
    memo_max_x = int(top_width * 0.45)
    memo_min_y = int(top_height * 0.08)
    memo_max_y = int(top_height * 0.7)
    memo_width  = memo_max_x - memo_min_x
    memo_height = memo_max_y - memo_min_y

    # Get the signature region bounds
    sig_min_x = int(top_width * 0.50)
    sig_max_x = int(top_width * 0.92)
    sig_min_y = int(top_height * 0.08)
    sig_max_y = int(top_height * 0.65)
    sig_width  = sig_max_x - sig_min_x
    sig_height = sig_max_y - sig_min_y

    # Get the bottom row
    acc_min_x = 0
    acc_max_x = width
    acc_min_y = int(height * 0.55)
    acc_max_y = height
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
    rout_field = (rout, image)

    acc = FieldData()
    acc.bounds = BoundingRect(acc_min_x, acc_min_y, acc_width, acc_height)
    acc.field_type = FieldType.FIELD_TYPE_ACCOUNT
    acc_field = (acc, image)

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


def extract_fields_entry_point(image_orig, image):

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
    lower_y_end   = height

    # draw the bounding region for visualization
    img_cpy = image.copy()

    # crop each section
    cropped_upper  = img_cpy[upper_y_start  : upper_y_end,  upper_x_start : upper_x_end]
    cropped_middle = img_cpy[middle_y_start : middle_y_end, upper_x_start : upper_x_end]
    cropped_lower  = img_cpy[lower_y_start  : lower_y_end,  upper_x_start : upper_x_end]

    upper_images  = upper_extract(cropped_upper) # BB WORKING
    middle_images = middle_extract(cropped_middle)
    lower_images  = lower_extract(cropped_lower)

    # compile the list of fields
    list = []
    for (field, img) in upper_images:
        field.bounds = BoundingRect(field.bounds.x + upper_x_start, 
                                    field.bounds.y + upper_y_start, 
                                    field.bounds.w, 
                                    field.bounds.h)

        list.append((field, img))
    
    for (field, img) in middle_images:
        field.bounds = BoundingRect(field.bounds.x + middle_x_start, 
                                    field.bounds.y + middle_y_start, 
                                    field.bounds.w, 
                                    field.bounds.h)

        list.append((field, img))
    
    for (field, img) in lower_images:
        field.bounds = BoundingRect(field.bounds.x + lower_x_start, 
                                    field.bounds.y + lower_y_start, 
                                    field.bounds.w, 
                                    field.bounds.h)

        list.append((field, img))

    return img_cpy, list
