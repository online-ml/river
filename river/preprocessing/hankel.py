from __future__ import annotations

from collections import deque
from typing import Literal

from river.base import Transformer

__all__ = ["Hankelizer"]


class Hankelizer(Transformer):
    """Time Delay Embedding using Hankelization.

    Convert a time series into a time delay embedded Hankel vectors.

    Args:
        w: The number of data snapshots to preserve
        return_partial: Whether to return partial Hankel matrices when the
            window is not full. Default "copy" fills missing with copies.

    Examples:
    >>> h = Hankelizer(w=3)
    >>> h.learn_one({"a": 1, "b": 2})
    >>> h.transform_one({"a": 1, "b": 2})
    {'a_0': 1, 'b_0': 2, 'a_1': 1, 'b_1': 2, 'a_2': 1, 'b_2': 2}

    >>> h = Hankelizer(w=3, return_partial=False)
    >>> h.learn_one({"a": 1, "b": 2})
    >>> h.transform_one({"a": 1, "b": 2})
    Traceback (most recent call last):
        ...
    ValueError: The window is not full yet. Set `return_partial` to True ...

    >>> h = Hankelizer(w=3, return_partial=True)
    >>> h.learn_one({"a": 1, "b": 2})
    >>> h.transform_one({"a": 1, "b": 2})
    {'a_0': nan, 'b_0': nan, 'a_1': nan, 'b_1': nan, 'a_2': 1, 'b_2': 2}

    Actually, transform_one does not care about the data as the learn should precede.
    >>> h.learn_one({"a": 3, "b": 4})
    >>> h.transform_one({"a": 5, "b": 6})
    {'a_0': nan, 'b_0': nan, 'a_1': 1, 'b_1': 2, 'a_2': 3, 'b_2': 4}
    >>> h._window
    deque([{'a': 1, 'b': 2}, {'a': 3, 'b': 4}], maxlen=3)

    Transform and learn in one go.
    >>> h.learn_transform_one({"a": 5, "b": 6})
    {'a_0': 1, 'b_0': 2, 'a_1': 3, 'b_1': 4, 'a_2': 5, 'b_2': 6}

    TODO:
        - [ ] Find out how to hankelize u while staying aligned with pipeline
    """

    def __init__(
        self, w: int = 2, return_partial: bool | Literal["copy"] = "copy"
    ):
        self.w = w
        self.return_partial = return_partial

        self._window: deque = deque(maxlen=self.w)
        self.feature_names_in_: list[str]
        self.n_features_in_: int

    def learn_one(self, x: dict):
        if not hasattr(self, "feature_names_in_") and isinstance(x, dict):
            self.feature_names_in_ = list(x.keys())
            self.n_features_in_ = len(x)

        self._window.append(x)

    def transform_one(self, x: dict):
        if not isinstance(x, dict):
            on_arrays = True
        else:
            on_arrays = False
        # TODO: If called before learn_one, creates duplicate sample
        _window = list(self._window)
        w_past_current = len(_window)
        if w_past_current == 0:
            _window = [x]
            # To avoid overflowing the window
            w_past_current = 1
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
        if on_arrays:
            import numpy as np

            return np.array([v for d in _window for v in d])
        else:
            return {
                f"{k}_{i}": v
                for i, d in enumerate(_window)
                for k, v in d.items()
            }

    def learn_transform_one(self, x: dict):
        self.learn_one(x)
        y = self.transform_one(x)
        return y
