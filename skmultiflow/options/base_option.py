__author__ = 'Guilherme Matsumoto'

from abc import ABCMeta, abstractmethod
from skmultiflow.core.base_object import BaseObject


class BaseOption(BaseObject, metaclass=ABCMeta):
    """Base Classifier class
    """
    def __init__(self):
        super().__init__()
        """ Initialization.
        """
        pass

    @abstractmethod
    def get_name(self):
        pass

    @abstractmethod
    def get_value(self):
        pass

    @abstractmethod
    def get_cli_char(self):
        pass

    @abstractmethod
    def get_option_type(self):
        pass

    @abstractmethod
    def set_value_via_cli_string(self, cli_string = None):
        pass

    @abstractmethod
    def get_cli_option_from_dictionary(self):
        pass

    def get_class_type(self):
        return 'option'

    @abstractmethod
    def get_info(self):
        pass