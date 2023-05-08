from __future__ import annotations

import abc
import math

from .utils import GradHess


class Loss(abc.ABC):
    """Base class to implement optimization objectives used in Stochastic Gradient Trees."""

    @abc.abstractmethod
    def compute_derivatives(self, y_true: float, y_pred: float) -> GradHess:
        """Return the gradient and hessian data concerning one instance and its prediction.

        Parameters
        ----------
        y_true
            Target value.
        y_pred
            Predicted target value.
        """
        raise NotImplementedError

    def transfer(self, y: float) -> float:
        """Optionally apply some transformation to the value predicted by the tree before
        returning it.

        For instance, in classification, the softmax operation might be applied.

        Parameters
        ----------
        y
            Value to be transformed.
        """
        return y


class BinaryCrossEntropyLoss(Loss):
    """Loss function used in binary classification tasks."""

    def compute_derivatives(self, y_true, y_pred):
        y_trs = self.transfer(y_pred)

        return GradHess(y_trs - y_true, y_trs * (1.0 - y_trs))

    def transfer(self, y):
        return 1.0 / (1.0 + math.exp(-y))


class SquaredErrorLoss(Loss):
    """Loss function used in regression tasks."""

    def compute_derivatives(self, y_true, y_pred):
        return GradHess(y_pred - y_true, 1.0)
