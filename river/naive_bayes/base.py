import abc
import math

import numpy as np
import pandas as pd
from scipy import sparse, special

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
        if jll.empty:
            return jll
        lse = pd.Series(special.logsumexp(jll, axis=1))
        return np.exp(jll.subtract(lse.values, axis="rows"))

    @property
    def _multiclass(self):
        return True


def from_dict(data: dict) -> pd.DataFrame:
    """Convert a dict into a pandas dataframe.
    This function is faster than pd.from_dict (01/02/2021).

    Parameters
    ----------
    data
        Input data as dict.

    Returns
    --------
        Dict to pandas dataframe.

    """
    dict_data, index = list(data.values()), list(data.keys())
    return pd.DataFrame(data=dict_data, index=index, dtype="float32")


def one_hot_encode(y: pd.Series) -> pd.DataFrame:
    """One hot encode input pandas series into sparse pandas DataFrame.

    Parameters
    ----------
    y
        Pandas Series of strings.

    Returns
    --------
    One hot encoded sparse dataframe.

    """
    classes = np.unique(y)
    indices = np.searchsorted(classes, y)
    indptr = np.hstack((0, np.cumsum(np.in1d(y, classes))))
    data = np.empty_like(indices)
    data.fill(1)
    return pd.DataFrame.sparse.from_spmatrix(
        sparse.csr_matrix(
            (data, indices, indptr), shape=(y.shape[0], classes.shape[0])
        ),
        index=y.index,
        columns=[str(c) for c in classes],
    )
