import abc
import math

import pandas as pd
import numpy as np

from scipy import special

from river import base


class BaseNB(base.Classifier):
    """Base Naive Bayes class."""

    @abc.abstractmethod
    def joint_log_likelihood(self, x: dict) -> float:
        """Compute the unnormalized posterior log-likelihood of x.

        The log-likelihood is `log P(c) + log P(x|c)`.

        """

    @abc.abstractmethod
    def joint_log_likelihood_many(self, X: pd.DataFrame) -> pd.DataFrame:
        """Compute the unnormalized posterior log-likelihood of x in mini-batches.

        The log-likelihood is `log P(c) + log P(x|c)`.

        """

    def predict_proba_one(self, x):
        """Return probabilities using the log-likelihoods."""
        jll = self.joint_log_likelihood(x)
        if not jll:
            return {}
        lse = special.logsumexp(list(jll.values()))
        return {label: math.exp(ll - lse) for label, ll in jll.items()}

    def predict_proba_many(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return probabilities using the log-likelihoods in mini-batchs setting."""
        jll = self.joint_log_likelihood_many(X)
        lse = pd.Series(special.logsumexp(jll, axis=1))
        return np.exp(jll.subtract(lse.values, axis="rows"))

    @property
    def _multiclass(self):
        return True


class Groupby:
    """Fast groupby for mini-batch.

    References:

        1. [Elizabeth Santorella, Fast groupby-apply operations in Python with and without Pandas](http://esantorella.com/2016/06/16/groupby/)
    """

    def __init__(self, keys):
        self.index, self.keys_as_int = np.unique(keys, return_inverse=True)
        self.n_keys = max(self.keys_as_int) + 1
        self.set_indices()

    def set_indices(self):
        self.indices = [[] for i in range(self.n_keys)]
        for i, k in enumerate(self.keys_as_int):
            self.indices[k].append(i)
        self.indices = [np.array(elt) for elt in self.indices]

    def apply(self, function, vector):
        result = []
        for k, idx in enumerate(self.indices):
            result.append(function(vector[idx], axis=0))

        return result, self.index
