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
    # pre_image, old_image = prp.preprocessEntryPoint(img)
    img, old_image = preprocessEntryPoint(img)

    ##################################################
    # FIELD EXTRACTTION PASS
    ##################################################
    # Returns a list of fields
    # img, fields = fe.extractFieldsEntryPoint(old_image, pre_image)
    nimg, fields = extract_fields_entry_point(old_image, img)

    if fields is None or len(fields) == 0:
        print("No fields were found!")
        return
    
    # Was the data preserved when returning?
    # Print and write to output file
    # count = 0
    # def_name = "resources/output/cropped_field"
    # for pair in fields:
    #     cv2.imshow('captcha_result', pair.image)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    #
    #     fname = def_name + str(count) + ".jpg";
    #     cv2.imwrite(fname , pair.image)
    #
    #     count += 1

    ##################################################
    # DATA EXTRACTION PASS
    ##################################################
    # for img, pair in fields:
    for (field, image) in fields:
        extract_data_entry_point(image, field)
        # de.extract_data_entry_point(image, field)
        
    ##################################################
    # POST PROCESS PASS
    ##################################################
    final_img = postprocessEntryPoint(old_image, dim, fields)

    # cv2.imshow('file img', final_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    json_dict = createJSONFromFieldDataList(fields)
    writeToJSONFile(json_dict, "out.json")

    return final_img, json_dict

