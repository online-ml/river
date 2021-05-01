import abc
import numbers
import typing

from . import schedulers


class Optimizer(abc.ABC):

    def __init__(self, lr: typing.Union[schedulers.Scheduler, numbers.Number]):
        if isinstance(lr, numbers.Number):
            lr = schedulers.Constant(lr)
        self.lr = lr
        self.n_iterations = 0

    @property
    def learning_rate(self) -> float:
        return self.lr.get(self.n_iterations)

    @classmethod
    def update_before_pred(self, w: dict) -> dict:
        """Updates a weight vector before a prediction is made."""
        return w

    def _update_after_pred(self, w: dict, g: dict, h: dict) -> dict:
        """Updates a weight vector given a gradient."""
        raise NotImplementedError

    def update_after_pred(self, w: dict, g: dict, h: dict) -> dict:

        # Update the weights
        w = self._update_after_pred(w, g, h)

        # Update the iteration counter
        self.n_iterations += 1

        return w

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return f'{self.__class__.__name__}({vars(self)})'
