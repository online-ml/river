__author__ = 'Guilherme Matsumoto'

from abc import ABCMeta, abstractmethod
from skmultiflow.core.base_object import BaseObject

class BaseInstance(BaseObject, metaclass=ABCMeta):
    '''
        Instance
        -------------------------------------------
        An instance read from the file
    '''

    def __init__(self):
        super().__init__()
        pass

    @abstractmethod
    def weight(self):
        pass

    @abstractmethod
    def set_weight(self):
        pass

    @abstractmethod
    def attribute(self, attIndex = -1):
        pass

    @abstractmethod
    def index_of(self, attribute = None):
        pass

    @abstractmethod
    def delete_attribute(self, attIndex = -1):
        pass

    @abstractmethod
    def insert_attribute(self, attIndex = -1):
        pass

    @abstractmethod
    def num_attribute(self):
        pass

    @abstractmethod
    def add_sparse_values(self, indexList = None, attValue = None, numberAttributes = -1):
        pass

    @abstractmethod
    def get_value(self, attIndex = -1):
        '''
            Each attribute type is assumed to be already known. 
        '''
        pass


    @abstractmethod
    def set_missing(self, attribute = None):
        pass

    @abstractmethod
    def set_value(self, attIndex = -1, newAttributeValue = None):
        '''
            It is expected that the correct attribute type will be passes.
            Flags need to be raised in case an incorrect attribute type is passed as parameter. 
        '''
        pass

    @abstractmethod
    def is_missing(self, attRef = None):
        '''
            Either int or attribute parameter. 
        '''
        pass

    @abstractmethod
    def is_missing_sparse(self, attRef = None):
        '''
            Used for sparse representation. Either int or attribute parameter. 
        '''
        pass

    @abstractmethod
    def index(self, arrayIndex = -1):
        pass

    @abstractmethod
    def value_sparse(selg, attIndex = -1):
        pass

    def get_class_type(self):
        return 'instance'

    @abstractmethod
    def get_info(self):
        pass