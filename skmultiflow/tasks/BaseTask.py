__author__ = 'Guilherme Matsumoto'

from abc import abstractmethod, ABCMeta

class BaseTask(metaclass=ABCMeta):

    def __init__(self):
        pass

    @abstractmethod
    def doTask(self, stream = None, classifier = None):
        pass