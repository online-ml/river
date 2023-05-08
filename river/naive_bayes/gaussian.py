from __future__ import annotations

import collections
import functools
import math

import pandas as pd
from scipy import special

from river import base, proba

__all__ = ["GaussianNB"]


class GaussianNB(base.Classifier):
    """Gaussian Naive Bayes.

    A Gaussian distribution $G_{cf}$ is maintained for each class $c$ and each feature $f$. Each
    Gaussian is updated using the amount associated with each feature; the details can be be found
    in `proba.Gaussian`. The joint log-likelihood is then obtained by summing the log probabilities
    of each feature associated with each class.

    Examples
    --------

    >>> from river import naive_bayes
    >>> from river import stream
    >>> import numpy as np

    >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    >>> Y = np.array([1, 1, 1, 2, 2, 2])

    >>> model = naive_bayes.GaussianNB()

    >>> for x, y in stream.iter_array(X, Y):
    ...     _ = model.learn_one(x, y)

    >>> model.predict_one({0: -0.8, 1: -1})
    1

    """

    def __init__(self):
        self.class_counts = collections.Counter()
        self.gaussians = collections.defaultdict(
            functools.partial(collections.defaultdict, proba.Gaussian)
        )

    def learn_one(self, x, y):
        self.class_counts.update((y,))

        for i, xi in x.items():
            self.gaussians[y][i].update(xi)

        return self

    def predict_proba_one(self, x):
        """Return probabilities using the log-likelihoods."""
        jll = self.joint_log_likelihood(x)
        if not jll:
            return {}
        lse = special.logsumexp(list(jll.values()))
        return {label: math.exp(ll - lse) for label, ll in jll.items()}

    def p_class(self, c):
        return self.class_counts[c] / sum(self.class_counts.values())

    def joint_log_likelihood(self, x):
        return {
            c: math.log(self.p_class(c))
            + sum(math.log(10e-10 + gaussians[i](xi)) for i, xi in x.items())
            for c, gaussians in self.gaussians.items()
        }

    def joint_log_likelihood_many(self, X: pd.DataFrame):
        pass

    @property
    def _multiclass(self):
        return True
