import backend.data_extraction.field.data.field_data as fd
import backend.preprocess.preprocess_main            as prp
import backend.field_extraction.field_extractor_main as fe
import backend.data_extraction.data_extraction_main  as de
import backend.postprocess.postprocess_main          as pop
import backend.json_utils.json_utils                 as ju
import os

import cv2

def test_subtraction(img):

    algo = "MOG2"
    if algo == 'MOG2':
        backSub = cv2.createBackgroundSubtractorMOG2()
    else:
        backSub = cv2.createBackgroundSubtractorKNN()
    
    fgMask = backSub.apply(img)
    cv2.rectangle(img, (10, 2), (100,20), (255,255,255), -1)

    cv2.imshow('Frame', img)
    cv2.imshow('FG Mask', fgMask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()  


    
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

    # cv2.imshow("Image", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()  

    ##################################################
    # PREPROCESS PASS
    ##################################################
    image = prp.preprocessEntryPoint(img)

    ##################################################
    # FIELD EXTRACTTION PASS
    ##################################################
    # Returns a list of fields
    img, fields = fe.extractFieldsEntryPoint(img, image)

    # cv2.imshow('captcha_result', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

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
    for pair in fields:
        de.extract_data_entry_point(pair)
        
    ##################################################
    # POST PROCESS PASS
    ##################################################
    pop.postprocessEntryPoint(image, fields)

    json_str = ju.createJSONFromFieldDataList(fields)
    ju.writeToJSONFile(json_str, "out.json")

    return img

def main():
    # image_file = "resources/images/simple_check.jpg"
    # filedir = os.path.abspath(os.path.dirname(__file__))
    # print(filedir)
    # image_file = os.path.join(filedir, '..\\resources\\images\\check_example.jpg')
    # image_file = "resources/images/check_example.jpg"
    # image_file = "resources/images/test_image.jpg"
    image_file = "resources/images/written_check.jpg"

    controller_entry_point(image_file)

if __name__ == "__main__":
    main()