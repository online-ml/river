from __future__ import annotations

import collections

import numpy as np

from river import optim, utils

__all__ = ["RMSProp"]


class RMSProp(optim.base.Optimizer):
    """RMSProp optimizer.

    Parameters
    ----------
    lr
    rho
    eps

    Examples
    --------

    >>> from river import datasets
    >>> from river import evaluate
    >>> from river import linear_model
    >>> from river import metrics
    >>> from river import optim
    >>> from river import preprocessing

    >>> dataset = datasets.Phishing()
    >>> optimizer = optim.RMSProp()
    >>> model = (
    ...     preprocessing.StandardScaler() |
    ...     linear_model.LogisticRegression(optimizer)
    ... )
    >>> metric = metrics.F1()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    F1: 87.24%

    References
    ----------
    [^1]: [Divide the gradient by a running average of itsrecent magnitude](https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)

    """

    def __init__(self, lr=0.1, rho=0.9, eps=1e-8):
        super().__init__(lr)
        self.rho = rho
        self.eps = eps
        self.g2 = None

    def _step_with_dict(self, w, g):
        if self.g2 is None:
            self.g2 = collections.defaultdict(float)

        for i, gi in g.items():
            self.g2[i] = self.rho * self.g2[i] + (1 - self.rho) * gi**2
            w[i] -= self.learning_rate / (self.g2[i] + self.eps) ** 0.5 * gi

        return w

    def _step_with_vector(self, w, g):
        if self.g2 is None:
            if isinstance(w, np.ndarray):
                self.g2 = np.zeros_like(w)
            else:
                self.g2 = utils.VectorDict()

        self.g2 = self.rho * self.g2 + (1 - self.rho) * g**2
        w -= self.learning_rate / (self.g2 + self.eps) ** 0.5 * g

        return w
