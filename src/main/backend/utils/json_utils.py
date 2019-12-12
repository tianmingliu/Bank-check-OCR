import json

"""
Writes a list to a specified file in JSON format.
 
fields is a tuple of (fields, image)

@param fields: list of fields of type FieldData

@return the JSON that was created
"""


def createJSONFromFieldDataList(fields):
    json_str = "{\n"
    id = 0
    for (pair, _) in fields:
        print(type(pair))
        json_str += str("\"element" + str(id) + "\": " + pair.to_json(indent=4) + ",\n")
        id += 1
    return json.loads(json_str[:-2] + "\n}\n")


"""
Writes the JSON to a file

@param json_str: the str containing JSON
@param filename: the name of the file to write the JSON to
"""


def writeToJSONFile(json_str, filename):
    fp = open(filename, 'w+')
    json.dump(json_str, fp, indent=4)
    fp.close()
