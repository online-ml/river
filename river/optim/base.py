from __future__ import annotations

import abc
import typing

import numpy as np

from river import base, optim, utils

__all__ = ["Initializer", "Scheduler", "Optimizer", "Loss"]

# Array-like weights/gradients: numpy arrays and `VectorDict`. These support elementwise
# arithmetic but not necessarily feature-name indexing (a raw `np.ndarray` does not).
VectorLike = typing.Union[utils.VectorDict, np.ndarray]  # noqa: UP007
# The "dict" path operates on feature-keyed, dict-like containers: plain dicts and `VectorDict`
# (which is dict-like). It explicitly excludes `np.ndarray`, which has no `.keys()`/`.items()` and
# cannot be indexed by feature name. `VectorDict` belongs to both unions because it implements both
# the mapping and the array protocols.
DictLike = typing.Union[dict, utils.VectorDict]  # noqa: UP007


class Initializer(base.Base, abc.ABC):
    """An initializer is used to set initial weights in a model."""

    @abc.abstractmethod
    def __call__(self, shape: int = 1) -> typing.Any:
        """Returns a fresh set of weights.

        The return type is shape-dependent (a scalar when ``shape == 1``, otherwise an array), and
        these values are routinely used as `collections.defaultdict` factories elsewhere, so the
        return is intentionally left dynamic.

        Parameters
        ----------
        shape
            Indicates how many weights to return. If `1`, then a single scalar value will be
            returned.

        """


class Scheduler(base.Base, abc.ABC):
    """Can be used to program the learning rate schedule of an `optim.base.Optimizer`."""

    @abc.abstractmethod
    def get(self, t: int) -> float:
        """Returns the learning rate at a given iteration.

        Parameters
        ----------
        t
            The iteration number.

        """

    def __repr__(self):
        return f"{self.__class__.__name__}({vars(self)})"


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

    def __init__(self, lr: int | float | Scheduler):
        if isinstance(lr, int) or isinstance(lr, float):
            lr = optim.schedulers.Constant(lr)
        self.lr = lr
        self.n_iterations = 0

    @property
    def _mutable_attributes(self):
        return {"lr"}

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

    def _step_with_dict(self, w: DictLike, g: DictLike) -> DictLike:
        raise NotImplementedError

    def _step_with_vector(self, w: VectorLike, g: VectorLike) -> VectorLike:
        raise NotImplementedError

    def step(self, w: dict | VectorLike, g: dict | VectorLike) -> dict | VectorLike:
        """Updates a weight vector given a gradient.

        Parameters
        ----------
        w
            A vector-like object containing weights. The weights are modified in-place.
        g
            A vector-like object of gradients.

        Returns
        -------
        The updated weights.

        """

        if (isinstance(w, utils.VectorDict) or isinstance(w, np.ndarray)) and (
            isinstance(g, utils.VectorDict) or isinstance(g, np.ndarray)
        ):
            w = self._step_with_vector(w, g)
            self.n_iterations += 1
            return w
        elif isinstance(w, dict) and isinstance(g, dict):
            w = self._step_with_dict(w, g)
            self.n_iterations += 1
            return w
        else:
            raise ValueError("Weights and gradients have incompatible types")

    def __repr__(self):
        return f"{self.__class__.__name__}({vars(self)})"


class Loss(base.Base, abc.ABC):
    """Base class for all loss functions."""

    def __repr__(self):
        return f"{self.__class__.__name__}({vars(self)})"

    @abc.abstractmethod
    def __call__(self, y_true, y_pred) -> typing.Any:
        """Returns the loss.

        Parameters
        ----------
        y_true
            Ground truth(s).
        y_pred
            Prediction(s).

        Returns
        -------
        The loss(es).

        """

    @abc.abstractmethod
    def gradient(self, y_true, y_pred) -> typing.Any:
        """Return the gradient with respect to y_pred.

        Parameters
        ----------
        y_true
            Ground truth(s).
        y_pred
            Prediction(s).

        Returns
        -------
        The gradient(s).

        """

    @abc.abstractmethod
    def mean_func(self, y_pred) -> typing.Any:
        """Mean function.

        This is the inverse of the link function. Typically, a loss function takes as input the raw
        output of a model. In the case of classification, the raw output would be logits. The mean
        function can be used to convert the raw output into a value that makes sense to the user,
        such as a probability.

        Parameters
        ----------
        y_pred
            Raw prediction(s).

        Returns
        -------
        The adjusted prediction(s).

        References
        ----------
        [^1]: [Wikipedia section on link and mean function](https://www.wikiwand.com/en/Generalized_linear_model#/Link_function)

        """
