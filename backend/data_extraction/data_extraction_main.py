import backend.data_extraction.field.data.field_data as field_data
import backend.data_extraction.field_list as field_list
import backend.data_extraction.digit_recognition.pyocr_ocr.handwriting_extract as data_extract
import backend.data_extraction.letter_recognition.src.main as hw_extract
from skimage.segmentation import clear_border
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2


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
def extract_data_entry_point(pair: field_data.DataPair):
    # Hard coded for now
    handwritten = True
    
    # Some struct that will contain the data 
    if handwritten:
        handwritten_extraction(pair)
    else:
        non_handwritten_extraction(pair)

    # Now identify the type of data
    # if not identify_extracted_field(pair):
    #     return False

    # Then validate
    #return validate_extracted_field(pair)
    return pair

"""
Performs the handwritten extraction from the provided image. If the
extraction was successful, field.data_info is filled out with the
extracted data.

@param image: image to extract the data from
@param field: a single field of type FieldData. 

@return True if the extraction was successful. False otherwise.
"""
def handwritten_extraction(pair: field_data.DataPair):
    # data = data_extract.extract_data(pair.image)
    # pair.data.extracted_data = data["text"]
    # pair.data.confidence = data["mean_conf"]
    print("Handwritten extraction: ")
    text = hw_extract.extract(pair.image)
    pair.data.extracted_data = text
    # print("\tExtracted data: " + pair.data.extracted_data)
    # print("\tMean confidence: " + str(pair.data.confidence))

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
Performs the non-handwritten extraction from the provided image. If the
extraction was successful, field.data_info is filled out with the
extracted data.

@param image: image to extract the data from
@param field: a single field of type FieldData. 

@return True if the extraction was successful. False otherwise.
"""


def account_routing_extraction(pair):
    print("Account/Writing extraction")
    # construct argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="path to input image")
    ap.add_argument("-r", "--reference", required=True, help="path to reference MICR E-13B font")
    args = vars(ap.parse_args())

    # init list of reference character names, in same order as they appear in reference
    # image where the digits, their names and:
    # T = Transit (delimit bank branch routing transit #)
    # U = On-us (delimit customer account number)
    # A = Amount (delimit transaction amount)
    # D = Dash (delimit parts of numbers, such as routing or account)
    charNames = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "T", "U", "A", "D"]

    # load ref MICR image, convert to grayscale and threshold it
    # this will cause digits to appear white on black background
    ref = cv2.imread(args["reference"])
    ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    ref = imutils.resize(ref, width=400)
    ref = cv2.threshold(ref, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # find contours in the MICR image and sort them left to right
    refCnts = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    refCnts = imutils.grab_contours(refCnts)
    refCnts = contours.sort_contours(refCnts, method="left-to-right")[0]

    """"" FROM PART 1, MAY BE REPLACED ENTIRELY
    # extract digits and symbols from list of contours, then
    # init a dict to map char name to ROI
    (refROIs, refLocs) = extract_digits_and_symbols(ref, refCnts, minW=10, minH=20)
    chars = {}

    # re-init the clone image so we can draw on it again
    clone = np.dstack([ref.copy()] * 3)

    # loop over reference ROIs and locations
    for (name, roi, loc) in zip(charNames, refROIs, refLocs):
        # draw bounding box surrounding character on output image
        (xA, yA, xB, yB) = loc
        cv2.rectangle(clone, (xA, yA), (xB, yB), (0, 255, 0), 2)

        # resize the ROI to a fixed size, then update the chars dict,
        # mapping char name to ROI
        roi = cv2.resize(roi, (36, 36))
        chars[name] = roi

        # display char ROI to ur screen
        cv2.imshow("Char", roi)
        cv2.waitKey(0)

    # show output of better method
    cv2.imshow("Better Method", clone)
    cv2.waitKey(0)
    """
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
    # to keep only the bottom 20% of the image (that's where the account info is)
    image = cv2.imread(args["image"])
    (h, w) = image.shape[:2]
    delta = int(h - (h * 0.2))
    bottom = image[delta:h, 0:w]

    # convert bottom image to grayscale, apply blackhat morphological operator
    # to find dark regions against a light background (the routing/account #s)
    gray = cv2.cvtColor(bottom, cv2.COLOR_BGR2GRAY)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)

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
    thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # remove any pixels that are touching borders of image (helps us in next
    # step when pruning contours)
    thresh = clear_border(thresh)

    # find contours in thresholded image, init list of group locations
    groupCnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
    groupLocs = sorted(groupLocs, key=lambda x:x[0])

    # loop over group locations
    for (gX, gY, gW, gH) in groupLocs:
        # init the group output of chars
        groupOutput = []

        # extract group ROI of chars from the grayscale image
        # then apply thresholding to segment the digits from background
        group = gray[gY - 5: gY + gH + 5, gX - 5: gX + gW + 5]
        group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        cv2.imshow("Group", group)
        cv2.waitKey(0)

        # find char contours in the group, then sort from left to right
        charCnts = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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

        # draw (padded) bounding box surrounding group along w/OCR output of group
        cv2.rectangle(image, (gX - 10, gY + delta - 10), (gX + gW + 10, gY + gY + delta), (0, 0, 255), 2)
        cv2.putText(image, "".join(groupOutput), (gX - 10, gY + delta - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 0, 255), 3)

        # add group output to overall check OCR output
        output.append("".join(groupOutput))

    # display output check OCR info to screen
    print("Check OCR: {}".format(" ".join(output)))
    cv2.imshow("Check OCR", image)
    cv2.waitKey(0)

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


def validate_extracted_field(pair):
    try:
        return field_list.GlobalFieldList[pair.data.field_type].validate()
    except KeyError:
        return False

    # for field in field_list.GlobalFieldList:
    #     if data.field_type == field.getType():
    #         return field.validate(data)
    # return False


"""
This function extracts the digits and symbols from the check image
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
                (sXA, sYA, sXB, sYB) = (np.inf, np.inf, -np.inf, -np.inf)  # init to positive and negative infinities

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
    extract_data_entry_point(None)
