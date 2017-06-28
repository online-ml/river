__author__ = 'Guilherme Matsumoto'

from skmultiflow.core.base_object import BaseObject
from abc import ABCMeta, abstractmethod


class BaseTransform(BaseObject, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()
        pass


    def get_class_type(self):
        return 'transform'

    @abstractmethod
    def get_info(self):
        pass

    @abstractmethod
    def transform(self, X):
        pass

    @abstractmethod
    def partial_fit_transform(self, X, y=None):
        pass

    @abstractmethod
    def partial_fit(self, X, y=None):
        pass



