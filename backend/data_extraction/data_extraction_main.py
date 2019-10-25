import backend.data_extraction.field.data.field_data as field_data
import backend.data_extraction.field_list as field_list
import backend.data_extraction.digit_recognition.pyocr_ocr.handwriting_extract as data_extract
import backend.data_extraction.letter_recognition.src.main as hw_extract

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
Scans the GlobalFieldList looking for a matching field using the data
found in field.data_info. If a match is found, then field.field_type is
set to the appriate FieldType.

@param field: a single field of type FieldData. 

@return True if the field was identified. False otherwise. 
"""
def identify_extracted_field(pair: field_data.DataPair):
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
def validate_extracted_field(pair: field_data.DataPair):
    try:
        return field_list.GlobalFieldList[pair.data.field_type].validate()
    except KeyError:
        return False

    # for field in field_list.GlobalFieldList:
    #     if data.field_type == field.getType():
    #         return field.validate(data)
    # return False
