__author__ = 'Guilherme Matsumoto'

from abc import ABCMeta, abstractmethod
from skmultiflow.core.base_object import BaseObject


class BaseEvaluator(BaseObject, metaclass=ABCMeta):
    """ BaseEvaluator

    The abstract class that works as a base model for all of this framework's 
    evaluators. It creates a basic interface that evaluation modules should 
    follow in order to use them with all the tools available in scikit-workflow.

    This class should not me instantiated, as none of its methods, except the 
    get_class_type, are implemented.

    Raises
    ------
    NotImplementedError: This is an abstract class.

    """

    def __init__(self):
        super().__init__()
        pass

    @abstractmethod
    def eval(self, stream, classifier):
        raise NotImplementedError

    @abstractmethod
    def partial_fit(self, X, y, classes=None):
        raise NotImplementedError

    @abstractmethod
    def predict(self, X):
        raise NotImplementedError

    def get_class_type(self):
        return 'evaluator'

    @abstractmethod
    def set_params(self, dict):
        raise NotImplementedError

    @abstractmethod
    def _update_metrics(self):
        raise NotImplementedError

    @abstractmethod
    def get_info(self):
        raise NotImplementedError
