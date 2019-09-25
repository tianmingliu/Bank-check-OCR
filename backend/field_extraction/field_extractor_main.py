import backend.data_extraction.field.data.field_data as field_data

"""
Provided an image of a check, a series of fields are
extracted from the image. For each field found, a FieldData
struct is created. This struct contains default values except
for the bounding box. The bounding box of the field is recorded
for further processing in a later pipelin stage.

@param image: Image of the check to extract fields from

@return A list of FieldData containg the bounding boxes of fields
extracted from the image.
"""
def extractFieldsEntryPoint(image):
    print("Extracting fields...")
    # do stuff

    # Just return a single data field for now
    list = [
        field_data.FieldData()
    ]

    # returns the data struct
    return list