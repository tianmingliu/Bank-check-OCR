import backend.data_extraction.field.data.field_data as field_data
import backend.data_extraction.field_list as field_list

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
def extractDataEntryPoint(image, field: field_data.FieldData):
    # Hard coded for now
    handwritten = True
    
    # Some struct that will contain the data 
    if (handwritten):
        handwrittenExtraction(image, field)
    else:
        nonHandwrittenExtraction(image, field)

    # Now identify the type of data
    if (not identifyExtractedField(field)):
        return False

    # Then validate
    return validateExtractedField(field)

"""
Performs the handwritten extraction from the provided image. If the
extraction was successful, field.data_info is filled out with the
extracted data.

@param image: image to extract the data from
@param field: a single field of type FieldData. 

@return True if the extraction was successful. False otherwise.
"""
def handwrittenExtraction(image, field: field_data.FieldData):
    print("Handwritten extraction")

"""
Performs the non-handwritten extraction from the provided image. If the
extraction was successful, field.data_info is filled out with the
extracted data.

@param image: image to extract the data from
@param field: a single field of type FieldData. 

@return True if the extraction was successful. False otherwise.
"""
def nonHandwrittenExtraction(image, field: field_data.FieldData):
    print("Non-handwritten extraction")

"""
Scans the GlobalFieldList looking for a matching field using the data
found in field.data_info. If a match is found, then field.field_type is
set to the appriate FieldType.

@param field: a single field of type FieldData. 

@return True if the field was identified. False otherwise. 
"""
def identifyExtractedField(data: field_data.FieldData):
    for field in field_list.GlobalFieldList:
        if field.identify(data):
            return True
    return False

"""
Validates the data found in field.data_info with its corresponding 
FieldType.

@param field: a single field of type FieldData. 

@return True if the field was valid. False otherwise. 
"""
def validateExtractedField(data: field_data.FieldData):
    for field in field_list.GlobalFieldList:
        if data.field_type == field.getType():
            return field.validate(data)
    return False