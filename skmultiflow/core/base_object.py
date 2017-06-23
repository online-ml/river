__author__ = 'Guilherme Matsumoto'

from abc import ABCMeta, abstractmethod

class BaseObject(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def get_class_type(self):
        pass

    @abstractmethod
    def get_info(self):
        pass