from .field.data.field_data import FieldData, FieldType
from .field_list import GlobalFieldList
from .extract_methods import extract_data_handwriting, extract_data_pytesseract, account_routing_extraction
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


def show(title, image):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def extract_data_entry_point(img, pair: FieldData):
    try:
        if pair.field_type == FieldType.FIELD_TYPE_ACCOUNT:
            account_routing_extraction(img, pair)
        if pair.field_type == FieldType.FIELD_TYPE_AMOUNT:
            handwritten_extraction(img, pair)
        elif pair.field_type == FieldType.FIELD_TYPE_AMOUNT_WRITTEN:
            handwritten_extraction(img, pair)
        elif pair.field_type == FieldType.FIELD_TYPE_DATE:
            handwritten_extraction(img, pair)
        elif pair.field_type == FieldType.FIELD_TYPE_MEMO:
            handwritten_extraction(img, pair)
        elif pair.field_type == FieldType.FIELD_TYPE_PAY_TO_ORDER_OF:
            handwritten_extraction(img, pair)
        elif pair.field_type == FieldType.FIELD_TYPE_ROUTING:
            account_routing_extraction(img, pair)
        elif pair.field_type == FieldType.FIELD_TYPE_SIGNATURE:
            handwritten_extraction(img, pair)
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
"""


def handwritten_extraction(image, pair: FieldData):
    text = extract_data_pytesseract(image)
    if text == "":
        text = extract_data_handwriting(image)
    pair.extracted_data = text


"""
Validates the data found in field.data_info with its corresponding 
FieldType.

@param field: a single field of type FieldData. 

@return True if the field was valid. False otherwise. 
"""


def validate_extracted_field(pair: FieldData):
    try:
        return GlobalFieldList[pair.field_type].validate(pair)
    except KeyError:
        return False

    # for field in field_list.GlobalFieldList:
    #     if data.field_type == field.getType():
    #         return field.validate(data)
    # return False


if __name__ == '__main__':
    extract_data_entry_point(None, None)
