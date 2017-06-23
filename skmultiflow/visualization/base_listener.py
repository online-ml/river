__author__ = 'Guilherme Matsumoto'

from abc import ABCMeta, abstractmethod
from skmultiflow.core.base_object import BaseObject


class BaseListener(BaseObject, metaclass=ABCMeta):

    def __init__(self):
        super().__init__()
        pass

    @abstractmethod
    def on_new_train_step(self, performance_point, train_step):
        pass

    def get_class_type(self):
        return 'listener'

    @abstractmethod
    def get_info(self):
        pass