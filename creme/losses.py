"""

- https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/lm/sgd_fast.pyx
"""
import abc
import math


__all__ = ['SquaredLoss', 'LogLoss']


class Loss(abc.ABC):

    @abc.abstractmethod
    def __call__(self, y_true, y_pred) -> float:
        pass

    @abc.abstractmethod
    def gradient(self, y_true, y_pred) -> dict:
        pass


class SquaredLoss(Loss):

    def __call__(self, y_true, y_pred):
        return (y_pred - y_true) ** 2

    def gradient(self, y_true, y_pred):
        return 2 * (y_pred - y_true)


class LogLoss(Loss):

    @staticmethod
    def clip_proba(p):
        return max(min(p, 1 - 1e-15), 1e-15)

    def __call__(self, y_true, y_pred):
        y_pred = self.clip_proba(y_pred)
        return -(y_true * math.log(y_pred) + (1 - y_true) * math.log(1 - y_pred))

    def gradient(self, y_true, y_pred):
        return self.clip_proba(y_pred) - y_true
