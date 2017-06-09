__author__ = 'Guilherme Matsumoto'

from abc import ABCMeta, abstractmethod


class BaseListener(metaclass=ABCMeta):

    def __init__(self):
        pass

    @abstractmethod
    def on_new_train_step(self, performance_point, train_step):
        pass
