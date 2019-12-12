from .field.data.field_data import FieldData, FieldType
from .field_list import GlobalFieldList
from .extract_methods import extract_data_handwriting, extract_data_pytesseract, account_routing_extraction
import cv2

"""
Creates a window displaying the image passed to it, making the title of the window = to the title parameter.
Also automatically sets window to display until keypress, and when keypress happens it destroys all windows it created

@param title: the title that the window should have
@param image: the image to be displayed
"""


def show(title, image):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


"""
Entry point for the Data Extraction stage of the pipeline.
Controls the overall flow of the program:
1. Checks which field type has been passed to it, and calls the appropriate data extraction method
2. Extracts the data from the provided field
3. Validates the extracted data using the Fields' validation methods

@param img: image to get the extracted data from
@param pair: FieldData object containing Field Type and in which to store extracted data

@return the pair FieldData object
"""


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
