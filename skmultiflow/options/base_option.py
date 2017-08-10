__author__ = 'Guilherme Matsumoto'

from abc import ABCMeta, abstractmethod
from skmultiflow.core.base_object import BaseObject


class BaseOption(BaseObject, metaclass=ABCMeta):
    """ BaseOption
    
    The abstract class that defines the constraints for all option 
    classes in this framework.
    
    Raises
    ------
    NotImplementedError: This is an abstract class.
    
    """

    def __init__(self):
        super().__init__()
        pass

    @abstractmethod
    def get_name(self):
        raise NotImplementedError

    @abstractmethod
    def get_value(self):
        raise NotImplementedError

    @abstractmethod
    def get_cli_char(self):
        raise NotImplementedError

    @abstractmethod
    def get_option_type(self):
        raise NotImplementedError

    def get_class_type(self):
        return 'option'

    @abstractmethod
    def get_info(self):
        raise NotImplementedError