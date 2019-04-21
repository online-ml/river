import abc
import typing

from . import lr_schedule


class Optimizer(abc.ABC):

    def __init__(self, lr: typing.Union[lr_schedule.LRScheduler, float]):
        self.lr = lr_schedule.ConstantLR(lr) if isinstance(lr, float) else lr
        self.n_iterations = 0

    @property
    def learning_rate(self) -> float:
        return self.lr.get(self.n_iterations)

    def update_before_pred(self, w: dict) -> dict:
        return w

    def _update_after_pred(self, w: dict, g: dict) -> dict:
        raise NotImplementedError

    def update_after_pred(self, w: dict, g: dict):

        # Update the weights
        w = self._update_after_pred(w, g)

        # Update the iteration counter
        self.n_iterations += 1

        return w
