import numbers
from typing import Union

import numpy as np

from river import base
from river import utils

from . import schedulers


VectorLike = Union[utils.VectorDict, np.ndarray]


class Optimizer(base.Base):
    """Optimizer interface.

    Every optimizer inherits from this base interface.

    Parameters
    ----------
    lr

    Attributes
    ----------
    learning_rate : float
        Returns the current learning rate value.

    """

    def __init__(self, lr: Union[schedulers.Scheduler, float]):
        if isinstance(lr, numbers.Number):
            lr = schedulers.Constant(lr)
        self.lr = lr
        self.n_iterations = 0

    @property
    def learning_rate(self) -> float:
        return self.lr.get(self.n_iterations)

    def look_ahead(self, w: dict) -> dict:
        """Updates a weight vector before a prediction is made.

        Parameters:
            w (dict): A dictionary of weight parameters. The weights are modified in-place.

        Returns:
            The updated weights.

        """
        return w

    def _step_with_dict(self, w: dict, g: dict) -> dict:
        raise NotImplementedError

    def _step_with_vector(self, w: VectorLike, g: VectorLike) -> VectorLike:
        raise NotImplementedError

    def step(
        self, w: Union[dict, VectorLike], g: Union[dict, VectorLike]
    ) -> Union[dict, VectorLike]:
        """Updates a weight vector given a gradient.

        Parameters:
            w (dict): A dictionary of weight parameters. The weights are modified in-place.
            g (dict): A dictionary of gradients.

        Returns:
            The updated weights.

        """

        # Update the weights
        # TODO: use functools.singledispatchmethod once support for Python 3.7 is dropped
        if isinstance(w, dict):
            w = self._step_with_dict(w, g)
        else:
            w = self._step_with_vector(w, g)

        # Update the iteration counter
        self.n_iterations += 1

        return w

    def __repr__(self):
        return f"{self.__class__.__name__}({vars(self)})"

    @classmethod
    def _unit_test_params(cls):
        return {}
