__author__ = 'Guilherme Matsumoto'

from abc import ABCMeta, abstractmethod

class BaseInstance(metaclass=ABCMeta):
    '''
        Instance
        -------------------------------------------
        An instance read from the file
    '''

    def __init__(self):
        pass

    @abstractmethod
    def weight(self):
        pass

    @abstractmethod
    def setWeight(self):
        pass

    @abstractmethod
    def attribute(self, attIndex = -1):
        pass

    @abstractmethod
    def indexOf(self, attribute = None):
        pass

    @abstractmethod
    def deleteAttribute(self, attIndex = -1):
        pass

    @abstractmethod
    def insertAttribute(self, attIndex = -1):
        pass

    @abstractmethod
    def numAttribute(self):
        pass

    @abstractmethod
    def addSparseValues(self, indexList = None, attValue = None, numberAttributes = -1):
        pass

    @abstractmethod
    def getValue(self, attIndex = -1):
        '''
            Each attribute type is assumed to be already known. 
        '''
        pass


    @abstractmethod
    def setMissing(self, attribute = None):
        pass

    @abstractmethod
    def setValue(self, attIndex = -1, newAttributeValue = None):
        '''
            It is expected that the correct attribute type will be passes.
            Flags need to be raised in case an incorrect attribute type is passed as parameter. 
        '''
        pass

    @abstractmethod
    def isMissing(self, attRef = None):
        '''
            Either int or attribute parameter. 
        '''
        pass

    @abstractmethod
    def isMissingSparse(self, attRef = None):
        '''
            Used for sparse representation. Either int or attribute parameter. 
        '''
        pass

    @abstractmethod
    def index(self, arrayIndex = -1):
        pass

    @abstractmethod
    def valueSparse(selg, attIndex = -1):
        pass

