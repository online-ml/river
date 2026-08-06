from __future__ import annotations

import collections

import numpy as np

from river import optim
from river.optim.base import DictLike

__all__ = ["NesterovMomentum"]


class NesterovMomentum(optim.base.Optimizer):
    """Nesterov Momentum optimizer.

    Parameters
    ----------
    lr
    rho

    Examples
    --------

    >>> from river import datasets
    >>> from river import evaluate
    >>> from river import linear_model
    >>> from river import metrics
    >>> from river import optim
    >>> from river import preprocessing

    >>> dataset = datasets.Phishing()
    >>> optimizer = optim.NesterovMomentum()
    >>> model = (
    ...     preprocessing.StandardScaler() |
    ...     linear_model.LogisticRegression(optimizer)
    ... )
    >>> metric = metrics.F1()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    F1: 84.22%

    """

    def __init__(self, lr=0.1, rho=0.9):
        super().__init__(lr)
        self.rho = rho
        self.s = collections.defaultdict(float)

    def look_ahead(self, w):
        # `w` may be a dict-like (iterating yields keys) or a NumPy array (iterate its indices).
        for i in range(len(w)) if isinstance(w, np.ndarray) else w:
            w[i] -= self.rho * self.s[i]

        return w

    def _step_with_dict(self, w: DictLike, g: DictLike) -> DictLike:
        # Move w back to it's initial position
        for i in range(len(w)) if isinstance(w, np.ndarray) else w:
            w[i] += self.rho * self.s[i]

        for i, gi in g.items():
            self.s[i] = self.rho * self.s[i] + self.learning_rate * gi
            w[i] -= self.s[i]

        return w
