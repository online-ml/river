__author__ = 'Guilherme Matsumoto'

from skmultiflow.options.BaseOption import BaseOption

class FileOption(BaseOption):
    '''
        fileOption class
        --------------------------------------
        Maintain options concerning file management.
    '''
    def __init__(self, optionType = None, optionName = None, optionValue = None, fileExtension = None, isOut = False):
        super().__init__()
        self.optionName = optionName
        self.optionValue = optionValue
        self.optionType = optionType
        self.fileType = fileExtension
        self.isOutput = isOut
        self.fileName = self.optionValue

    def getName(self):
        return self.optionName

    def getFileName(self):
        return self.fileName

    def getValue(self):
        return self.optionValue

    def getOptionType(self):
        return self.optionType

    def getCLIChar(self):
        return self.optionValue

    def isOutput(self):
        return self.isOutput

    def setValueViaCLIString(self, CLIstring = None):
        self.optionValue = CLIstring

    def getCLIOptionFromDictionary(self):
        return {"file_name" : "-n",
                "file_type" : "-t"}[self.optionType]