import abc
import backend.data_extraction.field.data.field_data as field_data

"""
Interface for identifying and validating a Field on a check.
There are three major functions:
1. identify: Identifies the data. If the data is of the instanced
             type, then it returns true.
2. validate: Validates the data. If the data is valid, it returns
             true.
3. getType: gets the type of the instanced variable
"""
class Field(abc.ABC):
    """
    Identifies the field passed to it. It can be assumed that data
    passed in a non-empty string. If the data is identified, field.field_type
    is set to the appropriate field type.

    @param field: a single field of type FieldData.

    @return True if the field was identified. False otherwise
    """
    @abc.abstractmethod
    def identify(self, field: field_data.DataPair):
        pass

    """
    Validates the field passed to it. It can be assumed the data passed
    is a non-empty string and has a valid FieldType.

    @param field: a single field of type FieldData.

    @return True if the field was valid. False otherwise
    """
    @abc.abstractmethod
    def validate(self, field: field_data.DataPair):
        pass

    """
    Retrieves the type of the instanced variable.

    @return FieldType of the instanced variable.
    """
    @abc.abstractmethod
    def getType(self):
        pass