import sys

import backend.data_extraction.field.data.field_data as field_data
import backend.data_extraction.field_list as field_list
# import backend.data_extraction.digit_recognition.pyocr_ocr.handwriting_extract as data_extract
# import backend.data_extraction.letter_recognition.src.main as hw_extract
import backend.preprocess.preprocess_main as prp
import backend.data_extraction.extract_methods as extract
from skimage.segmentation import clear_border
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
import pyocr
from PIL import Image
import os
import re

"""
Entry point for the Data Extraction stage of the pipeline.
Controls the overall flow of the program:
1. Checks if the input is handwritten or not (currently hardcoded)
2. Extracts the data from the provided field
3. identifies the extracted data using the GlobalFieldList
4. validates the extracted data using the GlobalFieldList

@param image: image to get the extracted data from
@param field: a single field of type FieldData who is filled out
during this stage of the pipeline. It can be assumed that a valid
bounding box has been set in the field.

@return True if the extraction was successful. False otherwise. 
"""


def show(title, image):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def extract_data_entry_point(img, pair: field_data.FieldData):
    try:
        if pair.field_type == field_data.FieldType.FIELD_TYPE_ACCOUNT:
            print("account type")
            account_routing_extraction(img, pair)
        if pair.field_type == field_data.FieldType.FIELD_TYPE_AMOUNT:
            # show("amount type", img)
            handwritten_extraction(img, pair)
        elif pair.field_type == field_data.FieldType.FIELD_TYPE_AMOUNT_WRITTEN:
            # show("amount written type", img)
            handwritten_extraction(img, pair)
        elif pair.field_type == field_data.FieldType.FIELD_TYPE_DATE:
            # show("date type", img)
            handwritten_extraction(img, pair)
        elif pair.field_type == field_data.FieldType.FIELD_TYPE_MEMO:
            print("memo type")
        elif pair.field_type == field_data.FieldType.FIELD_TYPE_PAY_TO_ORDER_OF:
            # show("pay to the order of type", img)
            handwritten_extraction(img, pair)
        elif pair.field_type == field_data.FieldType.FIELD_TYPE_ROUTING:
            print("routing type")
            account_routing_extraction(img, pair)
        elif pair.field_type == field_data.FieldType.FIELD_TYPE_SIGNATURE:
            print("signature type")
        elif pair.field_type == field_data.FieldType.FIELD_TYPE_NONE:
            print("none type")
        else:
            print("ERROR: Data extract: Invalid type.")
    except:
        None

    pair.validation = validate_extracted_field(pair)
    return pair


"""
Performs the handwritten extraction from the provided image. If the
extraction was successful, field.data_info is filled out with the
extracted data.

@param image: image to extract the data from
@param field: a single field of type FieldData. 

@return True if the extraction was successful. False otherwise.
"""


def handwritten_extraction(image, pair: field_data.FieldData):
    text = extract.extract_data_pytesseract(image)
    if text == "":
        text = extract.extract_data_handwriting(image)
    pair.extracted_data = text


"""
Performs the non-handwritten extraction from the provided image. If the
extraction was successful, field.data_info is filled out with the
extracted data.

@param image: image to extract the data from
@param field: a single field of type FieldData. 

@return True if the extraction was successful. False otherwise.
"""


def non_handwritten_extraction(pair: field_data.DataPair):
    print("Non-handwritten extraction")


"""
Performs account and routing extraction from the provided image. Checks the given
pair parameters' field_type field to see if it wants the routing or the account number, and then
sets the pair's extracted_data field to the accordingly, and returns the pair. If not successful, blank 
or garbage information is returned, otherwise both the extracted_data for routing and account would be a 
single string of digits.

@param img: image to extract the data from - this is a cropped version of full image, containing only the bottom 3rd
@param pair: the value that contains the type of the field that is requested, and the extracted_data itself to be returned

@return pair regardless of if extraction was successful; difference is only in the accuracy of pair.extracted_data
"""


def account_routing_extraction(img, pair: field_data.FieldData):
    print("Account/Writing extraction")
    if img is not None:
        filedir = os.path.abspath(os.path.dirname(__file__))
        ref_image_file = os.path.join(
            filedir, '..\\..\\resources\\images\\micr_e13b_reference.png')

        # init list of reference character names, in same order as they appear in reference
        # image where the digits, their names and:
        # T = Transit (delimit bank branch routing transit #)
        # U = On-us (delimit customer account number)
        # A = Amount (delimit transaction amount)
        # D = Dash (delimit parts of numbers, such as routing or account)
        charNames = ["1", "2", "3", "4", "5", "6",
                     "7", "8", "9", "0", "T", "U", "A", "D"]

        # load ref MICR image, convert to grayscale and threshold it
        # this will cause digits to appear white on black background
        ref = cv2.imread(ref_image_file)
        ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
        ref = imutils.resize(ref, width=400)
        ref = cv2.threshold(
            ref, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        # find contours in the MICR image and sort them left to right
        refCnts = cv2.findContours(
            ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        refCnts = imutils.grab_contours(refCnts)
        refCnts = contours.sort_contours(refCnts, method="left-to-right")[0]

        # extract digits and symbols from list of contours
        refROIs = extract_digits_and_symbols(ref, refCnts, minW=10, minH=20)[0]
        chars = {}

        # loop over reference ROIs
        for (name, roi) in zip(charNames, refROIs):
            # resize the ROI to a fixed size, then update the chars dict,
            # mapping char name to ROI
            roi = cv2.resize(roi, (36, 36))
            chars[name] = roi

        # init rectangular kernel along w/an empty list to store output of OCR
        rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 7))
        output = []

        # load the input image, grab its dimensions, and apply array slicing
        # to keep only the bottom 40% of the image (that's where the account/routing info is)
        (h, w) = img.shape[:2]
        delta = int(h - (h * 0.45))
        bottom = img[delta:h, 0:w]

        # convert bottom image to grayscale, apply blackhat morphological operator
        # to find dark regions against a light background (the routing/account #s)
        # gray = cv2.cvtColor(bottom, cv2.COLOR_BGR2GRAY)
        blackhat = cv2.morphologyEx(bottom, cv2.MORPH_BLACKHAT, rectKernel)

        # compute the Scharr gradient of the blackhat image, then scale
        # the rest back into the range [0, 255]
        gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        gradX = np.absolute(gradX)
        (minVal, maxVal) = (np.min(gradX), np.max(gradX))
        gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
        gradX = gradX.astype("uint8")

        # apply a closing operation using rectangular kernel to close gaps
        # between digits, then apply Otsus thresholding method to binarize image
        gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
        thresh = cv2.threshold(
            gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # remove any pixels that are touching borders of image (helps us in next
        # step when pruning contours)
        thresh = clear_border(thresh)

        # find contours in thresholded image, init list of group locations
        groupCnts = cv2.findContours(
            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        groupCnts = imutils.grab_contours(groupCnts)
        groupLocs = []

        # loop over group contours
        for (i, c) in enumerate(groupCnts):
            # compute bounding box of contour
            (x, y, w, h) = cv2.boundingRect(c)

            # only accept contour region as a grouping of chars if ROI sufficiently large
            if w > 50 and h > 15:
                groupLocs.append((x, y, w, h))

        # sort the digit locs from left to right
        groupLocs = sorted(groupLocs, key=lambda x: x[0])

        # loop over group locations
        for (gX, gY, gW, gH) in groupLocs:
            # init the group output of chars
            groupOutput = []

            # extract group ROI of chars from the grayscale image
            # then apply thresholding to segment the digits from background
            group = bottom[gY - 5: gY + gH + 5, gX - 5: gX + gW + 5]
            group = cv2.threshold(
                group, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

            # find char contours in the group, then sort from left to right
            charCnts = cv2.findContours(
                group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            charCnts = imutils.grab_contours(charCnts)
            charCnts = contours.sort_contours(charCnts, method="left-to-right")[0]

            # find chars and symbols in the group
            (rois, locs) = extract_digits_and_symbols(group, charCnts)

            # loop over ROIS from group
            for roi in rois:
                # init list of template matching scores and resize ROI to fixed size
                scores = []
                roi = cv2.resize(roi, (36, 36))

                # loop over ref char name and corresponding ROi
                for charName in charNames:
                    # apply correlation-based template matching, take score, update scores list
                    result = cv2.matchTemplate(roi, chars[charName], cv2.TM_CCOEFF)
                    (_, score, _, _) = cv2.minMaxLoc(result)
                    scores.append(score)

                # the classification for char ROI will be ref char name w/largest template matching score
                groupOutput.append(charNames[np.argmax(scores)])

            # add group output to overall check OCR output
            output.append("".join(groupOutput))

        # display output check OCR info to screen
        print("Check OCR: {}".format(" ".join(output)))

        if pair.field_type == field_data.FieldType.FIELD_TYPE_ROUTING:
            print('routing ' + output[0].translate({ord(c): None for c in 'TUAD'}))
            pair.extracted_data = output[0].translate({ord(c): None for c in 'TUAD'})
        elif pair.field_type == field_data.FieldType.FIELD_TYPE_ACCOUNT:
            print('account ' + output[1].translate({ord(c): None for c in 'TUAD'}))
            pair.extracted_data = output[1].translate({ord(c): None for c in 'TUAD'})
        return pair


"""
Scans the GlobalFieldList looking for a matching field using the data
found in field.data_info. If a match is found, then field.field_type is
set to the appriate FieldType.

@param field: a single field of type FieldData. 

@return True if the field was identified. False otherwise. 
"""


def identify_extracted_field(pair):
    for field in field_list.GlobalFieldList.values():
        if field.identify(pair.data):
            return True
    return False


"""
Validates the data found in field.data_info with its corresponding 
FieldType.

@param field: a single field of type FieldData. 

@return True if the field was valid. False otherwise. 
"""


def validate_extracted_field(pair: field_data.FieldData):
    try:
        return field_list.GlobalFieldList[pair.field_type].validate(pair)
    except KeyError:
        return False

    # for field in field_list.GlobalFieldList:
    #     if data.field_type == field.getType():
    #         return field.validate(data)
    # return False


"""
This function extracts each digit and symbol from the given image. If it is successful, it returns a tuple containing a
list of the roi (regions of interest, regions containing the chars to extract) and a list of locs (the actual locations
of those rois)

@param image: image to extract the data from - cropped version of full image, containing only an image of group of chars 
@param charCnts: list of character contours (what is used to determine each characters' location and identity)
@param minW: minimum width of a char for it to count as a character
@param minH: minimum height of a char for it to count as a character

@return tuple containing a list of rois and a list of locs
"""


def extract_digits_and_symbols(image, charCnts, minW=5, minH=15):
    # get Python iterator for character contours, and init ROI and location lists
    charIter = charCnts.__iter__()
    rois = []
    locs = []

    # loop over char contours until end of list
    while True:
        try:
            # get next char contour, compute bounding box, init ROI
            c = next(charIter)
            (cX, cY, cW, cH) = cv2.boundingRect(c)
            roi = None

            # check width/height if large enough, meaning we found a digit
            if cW >= minW and cH >= minH:
                # extract ROI
                roi = image[cY:cY + cH, cX: cX + cW]
                rois.append(roi)
                locs.append((cX, cY, cX + cW, cY + cH))
            else:  # otherwise it is a special symbol
                # MICR special symbols include 3 parts, so
                # need to get next 2 from iterator, then
                # init bounding box coordinates for symbol
                parts = [c, next(charIter), next(charIter)]
                # init to positive and negative infinities
                (sXA, sYA, sXB, sYB) = (np.inf, np.inf, -np.inf, -np.inf)

                # loop over parts
                for p in parts:
                    # calc bounding box for each part, update bookkeeping variables
                    (pX, pY, pW, pH) = cv2.boundingRect(p)
                    sXA = min(sXA, pX)
                    sYA = min(sYA, pY)
                    sXB = max(sXB, pX + pW)
                    sYB = max(sYB, pY + pH)

                # extract ROI
                roi = image[sYA:sYB, sXA:sXB]
                rois.append(roi)
                locs.append((sXA, sYA, sXB, sYB))
        except StopIteration:  # reached end of iterator, break from loop
            break

    # return tuple of ROIS and locations
    return rois, locs


if __name__ == '__main__':
    extract_data_entry_point(None, None)
