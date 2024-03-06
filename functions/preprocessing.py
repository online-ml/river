from collections import deque
from typing import Literal, Union

import numpy as np
from river.base import Transformer


def hankel(
    X: np.ndarray,
    hn: int,
    cut_rollover: bool = True,
) -> np.ndarray:
    """Create a Hankel matrix from a given input array.

    Args:
        X (np.ndarray): The input array.
        hn (int): The number of columns in the Hankel matrix.
        cut_rollover (bool, optional): Whether to cut the rollover part of the Hankel matrix. Defaults to True.

    Returns:
        np.ndarray: The Hankel matrix.

    TODO:
        - [ ] Add support for 2D arrays.

    Example:
    >>> X = np.array([1., 2., 3., 4., 5.])
    >>> hankel(X, 3, cut_rollover=False)
    array([[1., 2., 3.],
           [2., 3., 4.],
           [3., 4., 5.],
           [4., 5., 1.],
           [5., 1., 2.]])
    >>> hankel(X, 3)
    array([[1., 2., 3.],
           [2., 3., 4.],
           [3., 4., 5.]])
    """
    hX = np.empty((X.shape[0], hn))
    for i in range(hn):
        hX[:, i] = X
        X = np.roll(X, -1)
    if cut_rollover:
        hX = hX[: -hn + 1]
    return hX


class Hankelizer(Transformer):
    """Time Delay Embedding using Hankelization.

    Convert a time series into a time delay embedded Hankel vectors.

    Args:
        w: The number of data snapshots to preserve
        return_partial: Whether to return partial Hankel matrices when the
            window is not full. Default "copy" fills missing with copies.

    Examples:
    >>> h = Hankelizer(w=3)
    >>> h.transform_one({"a": 1, "b": 2})
    {'a_0': 1, 'b_0': 2, 'a_1': 1, 'b_1': 2, 'a_2': 1, 'b_2': 2}

    >>> h = Hankelizer(w=3, return_partial=False)
    >>> h.transform_one({"a": 1, "b": 2})
    Traceback (most recent call last):
        ...
    ValueError: The window is not full yet. Set `return_partial` to True ...

    >>> h = Hankelizer(w=3, return_partial=True)
    >>> h.transform_one({"a": 1, "b": 2})
    {'a_0': nan, 'b_0': nan, 'a_1': nan, 'b_1': nan, 'a_2': 1, 'b_2': 2}

    Transformation is stateless so we lost previous data.
    >>> h.transform_one({"a": 3, "b": 4})
    {'a_0': nan, 'b_0': nan, 'a_1': nan, 'b_1': nan, 'a_2': 3, 'b_2': 4}
    >>> h._window
    deque([], maxlen=2)
    >>> h.learn_one({"a": 1, "b": 2})

    Transform and learn in one go.
    >>> h.learn_transform_one({"a": 3, "b": 4})
    {'a_0': nan, 'b_0': nan, 'a_1': 1, 'b_1': 2, 'a_2': 3, 'b_2': 4}
    >>> h.transform_one({"a": 5, "b": 6})
    {'a_0': 1, 'b_0': 2, 'a_1': 3, 'b_1': 4, 'a_2': 5, 'b_2': 6}
    """

    def __init__(
        self, w: int, return_partial: Union[bool, Literal["copy"]] = "copy"
    ):
        self.w = w
        self.return_partial = return_partial

        self._window = deque(maxlen=self.w - 1)
        self.feature_names_in_: list[str]
        self.n_features_in_: int

    def learn_one(self, x: dict):
        if not hasattr(self, "feature_names_in_"):
            self.feature_names_in_ = list(x.keys())
            self.n_features_in_ = len(x)

        self._window.append(x)

    def transform_one(self, x: dict):
        _window = list(self._window) + [x]
        w_past_current = len(_window)
        if not self.return_partial and w_past_current < self.w:
            raise ValueError(
                "The window is not full yet. Set `return_partial` to True to return partial Hankel matrices."
            )
        else:
            n_missing = self.w - w_past_current
            _window = [_window[0]] * (n_missing) + _window
            if not self.return_partial == "copy":
                for i in range(n_missing):
                    _window[i] = {k: float("nan") for k in _window[0]}
            return {
                f"{k}_{i}": v
                for i, d in enumerate(_window)
                for k, v in d.items()
            }

    def learn_transform_one(self, x: dict):
        y = self.transform_one(x)
        self.learn_one(x)
        return y
