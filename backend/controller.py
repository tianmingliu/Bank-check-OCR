import backend.preprocess.preprocess_main            as prp
import backend.field_extraction.field_extractor_main as fe
import backend.data_extraction.data_extraction_main  as de
import backend.postprocess.postprocess_main          as pop

"""
Writes a list to a specified file in JSON format.

@param fields: list of fields of type FieldData
@param filename: file to write to
"""
def writeToJSONOutput(fields, filename): 
    fp = open(filename, 'w+')
    fp.write("{\n")

    for field in fields:
        fp.write(str(field))

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

    writeToJSONOutput(fields, "out.json")

if __name__ == "__main__":
    main()