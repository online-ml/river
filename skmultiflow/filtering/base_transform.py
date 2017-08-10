__author__ = 'Guilherme Matsumoto'

from skmultiflow.core.base_object import BaseObject
from abc import ABCMeta, abstractmethod


class BaseTransform(BaseObject, metaclass=ABCMeta):
    """ BaseTransform
    
    Abstract class that explicits the constraints to all Transform objects 
    in this framework.
    
    This class should not be instantiated as it has no implementations and 
    will raise NotImplementedErrors.
    
    Raises
    ------
    NotImplementedError: This is an abstract class.
    
    """

    def __init__(self):
        super().__init__()

    def get_class_type(self):
        return 'transform'

    @abstractmethod
    def get_info(self):
        raise NotImplementedError

    @abstractmethod
    def transform(self, X):
        raise NotImplementedError

    @abstractmethod
    def partial_fit_transform(self, X, y=None):
        raise NotImplementedError

    @abstractmethod
    def partial_fit(self, X, y=None):
        raise NotImplementedError



