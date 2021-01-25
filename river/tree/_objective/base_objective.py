from abc import ABCMeta

from .._utils import GradHess


class BaseObjective(metaclass=ABCMeta):
    """ Base class to implement optimization objectives used in Streaming Gradient Trees. """

    def compute_derivatives(self, y: float, y_pred: float) -> GradHess:
        """ Return the gradient and hessian data concerning one instance and its prediction.

        Parameters
        ----------
        y
            Target value.
        y_pred
            Predicted target value.
        """
        pass

    def transfer(self, y: float) -> float:
        """ Optionally apply some transformation to the value predicted by the tree before
        returning them.

        For instance, in classification, the softmax operation might be applied.

        Parameters
        ----------
        y
            Value to be transformed.
        """
        return y
