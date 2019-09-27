import backend.data_extraction.field.data.field_data as fd
import backend.preprocess.preprocess_main            as prp
import backend.field_extraction.field_extractor_main as fe
import backend.data_extraction.data_extraction_main  as de
import backend.postprocess.postprocess_main          as pop
import backend.json_utils.json_utils                 as ju

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
    # Read RGB image as greyscale
    img = cv2.imread(image_file)  

    cv2.imshow("Canny edge detection", img)
    cv2.waitKey(0)

    # Destroying present windows on screen 
    cv2.destroyAllWindows()  

    ##################################################
    # PREPROCESS PASS
    ##################################################
    image = prp.preprocessEntryPoint(img)

    ##################################################
    # FIELD EXTRACTTION PASS
    ##################################################
    # Returns a list of fields
    fields = fe.extractFieldsEntryPoint(image)
    if fields is None or len(fields) == 0:
        print("No fields were found!")
        return

    ##################################################
    # DATA EXTRACTION PASS
    ##################################################
    for field in fields:
        de.extractDataEntryPoint(image, field)

    ##################################################
    # DATA EXTRACTION PASS
    ##################################################
    pop.postprocessEntryPoint(image, fields)

    json_str = ju.createJSONFromFieldDataList(fields)
    ju.writeToJSONFile(json_str, "out.json")

def main():
    # image_file = "resources/images/simple_check.jpg"
    image_file = "resources/images/check_example.jpg"

    controller_entry_point(image_file)

if __name__ == "__main__":
    main()