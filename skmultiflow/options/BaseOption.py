__author__ = 'Guilherme Matsumoto'

from abc import ABCMeta, abstractmethod


class BaseOption(metaclass=ABCMeta):
    """Base Classifier class
    """
    def __init__(self):
        """ Initialization.
        """
        pass

    @abstractmethod
    def getName(self):
        pass

    @abstractmethod
    def getValue(self):
        pass

    @abstractmethod
    def getCLIChar(self):
        pass

    @abstractmethod
    def getOptionType(self):
        pass

    @abstractmethod
    def setValueViaCLIString(self, CLIstring = None):
        pass

    @abstractmethod
    def getCLIOptionFromDictionary(self):
        pass
