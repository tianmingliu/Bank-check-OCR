import backend.data_extraction.field.data.field_data as fd

from dataclasses_json import dataclass_json
from typing import List

"""
Writes a list to a specified file in JSON format.
 
fields is a tuple of (fields, image)

@param fields: list of fields of type FieldData
@param filename: file to write to
"""
# def createJSONFromFieldDataList(fields: List[fd.DataPair]):
def createJSONFromFieldDataList(fields):
    json_str = ""
    id = 0
    for (pair, _) in fields:
        print(type(pair))
        json_str += str("\"element" + str(id) + "\": " + pair.to_json(indent=4) + ",\n")
        id += 1
    return json_str[:-2] + "\n"

def writeToJSONFile(json_str, filename): 
    fp = open(filename, 'w+')
    fp.write("{\n")
    fp.write(json_str)
    fp.write("}\n")
    fp.close()