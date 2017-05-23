__author__ = 'Guilherme Matsumoto'

from skmultiflow.options.BaseOption import BaseOption

class NumericOption(BaseOption):
    def __init__(self, optionType = "NUM", optionName = None, optionCLIParser = None, optionValue = None):
        super().__init__()
        self.optionName = optionName
        self.optionCLIParser = optionCLIParser
        self.optionValue = optionValue
        self.optionType = optionType
        pass

    def getName(self):
        return self.optionName

    def getValue(self):
        return self.optionValue

    def getCLIChar(self):
        return self.optionCLIParser

    def getOptionType(self):
        return self.optionType

    def setValueViaCLIString(self, CLIstring = None):
        pass

    def getCLIOptionFromDictionary(self):
        pass
