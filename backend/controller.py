import backend.data_extraction.field.data.field_data as fd
import backend.preprocess.preprocess_main            as prp
import backend.field_extraction.field_extractor_main as fe
import backend.data_extraction.data_extraction_main  as de
import backend.postprocess.postprocess_main          as pop

from dataclasses_json import dataclass_json

from typing import List


"""
Writes a list to a specified file in JSON format.
 
@param fields: list of fields of type FieldData
@param filename: file to write to
"""
def createJSONFromFieldDataList(fields: List[fd.FieldData]):
    json_str = ""
    id = 0
    for field in fields:
        json_str += str("\"element" + str(id) + "\": " + field.to_json(indent=4) + ",\n")
        id += 1
    return json_str[:-2] + "\n"

def writeToJSONFile(json_str, filename): 
    fp = open(filename, 'w+')
    fp.write("{\n")
    fp.write(json_str)
    fp.write("}\n")
    fp.close()
    
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
def main():
    image = None

    prp.preprocessEntryPoint(image)

    # Returns a list of fields
    fields = fe.extractFieldsEntryPoint(image)
    if fields is None or len(fields) == 0:
        print("No fields were found!")
        return

    for field in fields:
        de.extractDataEntryPoint(image, field)

    pop.postprocessEntryPoint(image, fields)

    json_str = createJSONFromFieldDataList(fields)
    writeToJSONFile(json_str, "out.json")

if __name__ == "__main__":
    main()