import backend.data_extraction.field.data.field_data as field_data

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

        # extract the actual padded ROI
        # roi = orig[startY:endY, startX:endX]

        # in order to apply Tesseract v4 to OCR text we must supply
        # (1) a language, (2) an OEM flag of 4, indicating that the we
        # wish to use the LSTM neural net model for OCR, and finally
        # (3) an OEM value, in this case, 7 which implies that we are
        # treating the ROI as a single line of text
        # config = ("-l eng --oem 1 --psm 7")
        text = "" #pytesseract.image_to_string(roi, config=config)

        # add the bounding box coordinates and OCR'd text to the list
        # of results
        results.append(((startX, startY, endX, endY), text))

    # sort the results bounding box coordinates from top to bottom
    results = sorted(results, key=lambda r:r[0][1])

    print("Number of boxes: " + str(len(results)))

    # check for collisions between the boxes
    new_results = []
    for ((a_start_x, a_start_y, a_end_x, a_end_y), _) in results:
        print("run")

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
        # display the text OCR'd by Tesseract
        print("OCR TEXT")
        print("========")

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
    
    # END ORIGINAL

    
    # cv2.imshow('captcha_result', img_cpy)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

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


"""
For the following processing functions:
@param image a cropped image that requires special processing
@return a list of cropped images

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

    detect_text(cropped_right, .5, .1)


def process_middle_region(image):
    height = image.shape[0]
    width  = image.shape[1]

    dim_h = int(height / 3)

    top_y = 0
    lower_y = top_y + dim_h - 20

    cropped_upper = image[top_y : top_y + dim_h, 0 : width]
    cropped_lower = image[lower_y : height, 0 : width]

    # show(cropped_upper, "Upper Middle")
    # show(cropped_lower, "Lower Middle")

    iso_upper = isolate_text(cropped_upper)
    iso_lower = isolate_text(cropped_lower)

    # detect_text(iso_upper, .8, .4)
    # detect_text(iso_lower, .8, .4)


def process_lower_region(image):
    height = image.shape[0]
    width  = image.shape[1]

    iso_image = isolate_text(image)

    # detect_text(iso_image, .1, .1)

    # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 3))
    # dilated = cv2.dilate(iso_image, kernel, iterations=3)

    # show(dilated, "Dilated img")

    # contours, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # new_results = []
    # for contour in contours:
    #     [x, y, w, h] = cv2.boundingRect(contour)

    #     has_collision = False
    #     idx = 0
    #     for (a_start_x, a_start_y, a_end_x, a_end_y) in new_results:
    #         if check_bounding_collision((a_start_x, a_start_y, a_end_x, a_end_y), (x, y, x + w, y + h)):
    #             has_collision = True
    #             break
    #         else:
    #             idx += 1

    #     if has_collision:
    #         # determine new max/min
    #         min_x = 0
    #         min_y = 0
    #         max_x = 0
    #         max_y = 0

    #         if a_start_x <= x:
    #             min_x = a_start_x
    #         else:
    #             min_x = x
            
    #         if a_start_y <= y:
    #             min_y = a_start_y
    #         else:
    #             min_y = y

    #         if a_end_x >= x + w:
    #             max_x = a_end_x
    #         else:
    #             max_x = x + w
            
    #         if a_end_y >= y + h:
    #             max_y = a_end_y
    #         else:
    #             max_y = y + h

    #         # set the new bounding box
    #         new_results[idx] = (min_x, min_y, max_x, max_y)
    #     else:
    #         minx = x
    #         miny = y
    #         maxx = x + w
    #         maxy = y + h 

    #         xthresh = .8
    #         ythresh = .8

    #         ratiox = ((maxx - minx) / width)
    #         ratioy = ((maxy - miny) / height)
    #         if not (ratiox >= xthresh and ratioy >= ythresh):
    #             new_results.append((x, y, x + w, y + h))
    #         else:
    #             print("Very large box! ratiox: " + str(ratiox) + " ratioy: " + str(ratioy))
    #             print("Image dim: " + str(width) + " " + str(height))
    #             print("Cropped dim: " + str(maxx - minx) + " " + str((maxy - miny)))
        

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

    cropped_upper  = img_cpy[upper_y  : upper_y  + dim_h, upper_x  : width]
    cropped_middle = img_cpy[middle_y : middle_y + dim_h, middle_x : width]
    cropped_lower  = img_cpy[lower_y  : lower_y  + dim_h, lower_x  : width]

    process_upper_region(cropped_upper)
    process_middle_region(cropped_middle)
    process_lower_region(cropped_lower)

    list = []

    return img_cpy, list
