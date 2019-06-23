import abc

from ... import utils


class Loss(abc.ABC):

    @abc.abstractmethod
    def __call__(self, y_true, y_pred) -> float:
        """Returns the loss."""

    @abc.abstractmethod
    def gradient(self, y_true, y_pred) -> float:
        """Returns the gradient with respect to ``y_pred``."""


class ClassificationLoss(Loss):

    @staticmethod
    def clamp_proba(p):
        return utils.clamp(p, minimum=1e-15, maximum=1 - 1e-15)


class BinaryClassificationLoss(ClassificationLoss):
    """A loss appropriate binary classification tasks."""


class MultiClassificationLoss(ClassificationLoss):
    """A loss appropriate for multi-class classification tasks."""


class RegressionLoss(Loss):
    """A loss appropriate for regression tasks."""
