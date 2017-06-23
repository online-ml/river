__author__ = 'Guilherme Matsumoto'
from abc import ABCMeta, abstractmethod
from skmultiflow.core.base_object import BaseObject


class BaseEvaluator(BaseObject, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()
        pass

    @abstractmethod
    def eval(self, stream, classifier):
        pass

    @abstractmethod
    def partial_fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    def get_class_type(self):
        return 'evaluator'

    @abstractmethod
    def set_params(self, dict):
        pass

    @abstractmethod
    def update_metrics(self):
        pass

    @abstractmethod
    def get_info(self):
        pass