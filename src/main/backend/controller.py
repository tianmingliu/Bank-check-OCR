from .preprocess.preprocess_main import preprocessEntryPoint
from .field_extraction.field_extractor_main import extract_fields_entry_point 
from .data_extraction.data_extraction_main import extract_data_entry_point
from .postprocess.postprocess_main import postprocessEntryPoint
from .utils.json_utils import createJSONFromFieldDataList, writeToJSONFile

import cv2
    
"""
Current entry point for the program.

Controls the overall pipeline:
1. Preprocess
2. Extract all fields
3. Extract data from each field
4. Validate data
5. Postprocess
6. Write to JSON file
7. Send postprocess data to the frontend 


@param image_file: file containing path to the image

@return (final_image, json_dict) tuple containing the final image after all processing, and the JSON dictionary object
"""


def controller_entry_point(image_file):

    print(image_file)

    # Read RGB image as greyscale
    img = cv2.imread(image_file)  

    ##################################################
    # PREPROCESS PASS
    ##################################################
    # Save the original dimensions
    height = img.shape[0]
    width  = img.shape[1]
    dim = (width, height)

    # Process the image
    img, old_image = preprocessEntryPoint(img)

    ##################################################
    # FIELD EXTRACTION PASS
    ##################################################
    # Returns a list of fields
    nimg, fields = extract_fields_entry_point(old_image, img)

    if fields is None or len(fields) == 0:
        print("No fields were found!")
        return

    ##################################################
    # DATA EXTRACTION PASS
    ##################################################
    # for img, pair in fields:
    for (field, image) in fields:
        extract_data_entry_point(image, field)
        
    ##################################################
    # POST PROCESS PASS
    ##################################################
    final_img = postprocessEntryPoint(old_image, dim, fields)

    json_dict = createJSONFromFieldDataList(fields)
    writeToJSONFile(json_dict, "out.json")

    return final_img, json_dict

