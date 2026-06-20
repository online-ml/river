from __future__ import annotations

import numpy as np

from river import base, optim, utils
from river.optim.base import DictLike

__all__ = ["Newton"]


class Newton(optim.base.Optimizer):
    """Online Newton Step (ONS) optimizer.

    This optimizer uses second-order information (i.e. the Hessian of the cost function) in
    addition to first-order information (i.e. the gradient of the cost function). It maintains the
    matrix

    $$A_t = \\epsilon I + \\sum_{s=1}^{t} g_s g_s^\\intercal$$

    where $g_s$ is the gradient at step $s$, and applies the update

    $$w_{t+1} = w_t - \\eta A_t^{-1} g_t.$$

    The inverse $A_t^{-1}$ is never computed from scratch. Instead it is updated in-place at each
    step with the Sherman-Morrison formula (see `utils.math.sherman_morrison`), which turns the
    rank-1 update $A_t = A_{t-1} + g_t g_t^\\intercal$ into a single rank-1 update of the inverse.
    This costs $O(d^2)$ time and memory, where $d$ is the number of features seen so far.

    Parameters
    ----------
    lr
        Learning rate. This corresponds to $1 / \\gamma$ in the paper.
    eps
        Regularization term used to initialize the Hessian to $\\epsilon I$. Its inverse
        $1 / \\epsilon$ is used as the starting value of $A_t^{-1}$, so a smaller `eps` allows
        larger steps early on.

    Examples
    --------

    >>> from river import datasets
    >>> from river import evaluate
    >>> from river import linear_model
    >>> from river import metrics
    >>> from river import optim
    >>> from river import preprocessing

    >>> dataset = datasets.Phishing()
    >>> optimizer = optim.Newton()
    >>> model = (
    ...     preprocessing.StandardScaler() |
    ...     linear_model.LogisticRegression(optimizer)
    ... )
    >>> metric = metrics.F1()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    F1: 73.62%

    References
    ----------
    [^1]: [Hazan, E., Agarwal, A. and Kale, S., 2007. Logarithmic regret algorithms for online convex optimization. Machine Learning, 69(2-3), pp.169-192](https://www.cs.princeton.edu/~ehazan/papers/log-journal.pdf)
    [^2]: [Hazan, E., 2016. Introduction to online convex optimization. Foundations and Trends in Optimization, 2(3-4), pp.157-325 (Chapter 4 states the ONS regret theorem)](https://arxiv.org/abs/1909.05207)
    [^3]: [Hazan, E., Koren, T. and Levy, K.Y., 2014. Logistic regression: Tight bounds for stochastic and online optimization. COLT 2014 — shows the exp-concavity constant of the logistic loss is e^{-D}, so ONS has no asymptotic edge over first-order methods on logistic regression until T is exponentially large](https://proceedings.mlr.press/v35/hazan14a.html)

    """

    def __init__(self, lr=0.1, eps=1e-5):
        super().__init__(lr)
        self.eps = eps
        self._idx: dict[base.typing.FeatureName, int] = {}
        # `_H_inv` is the inverse of the accumulated Hessian A = eps * I + sum_t g_t g_t^T. It is
        # maintained in-place via a Sherman-Morrison rank-1 update (one BLAS dger per step), hence
        # the Fortran ordering. The implied A_0 = eps * I, so A_0^{-1} = (1 / eps) * I.
        self._H_inv = np.zeros((0, 0), dtype=np.float64, order="F")
        self._cap = 0

    def _grow(self, needed: int) -> None:
        new_cap = max(needed, max(8, self._cap * 2))
        new_H_inv = np.eye(new_cap, dtype=np.float64, order="F") / self.eps
        if self._cap:
            new_H_inv[: self._cap, : self._cap] = self._H_inv
        self._H_inv = new_H_inv
        self._cap = new_cap

    def _ensure_features(self, features) -> None:
        idx = self._idx
        for f in features:
            if f not in idx:
                idx[f] = len(idx)
        if len(idx) > self._cap:
            self._grow(len(idx))

    def _step_with_dict(self, w: DictLike, g: DictLike) -> DictLike:
        self._ensure_features(g.keys())
        g_arr = np.zeros(self._cap, dtype=np.float64)
        for f, gi in g.items():
            g_arr[self._idx[f]] = gi

        # Rank-1 update of the inverse Hessian: A_t = A_{t-1} + g_t g_t^T.
        utils.math.sherman_morrison(A=self._H_inv, u=g_arr, v=g_arr)

        # Online Newton step: w <- w - lr * A_t^{-1} g_t. The step has support over every feature
        # seen so far, because A_t^{-1} couples features through their gradient correlations.
        step = self._H_inv @ g_arr
        lr = self.learning_rate
        for f, i in self._idx.items():
            w[f] -= lr * step[i]

        return w
