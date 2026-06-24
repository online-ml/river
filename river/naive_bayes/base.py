from __future__ import annotations

import abc
import math
import typing

import numpy as np
from scipy import sparse, special

from river import base, utils

if typing.TYPE_CHECKING:
    import pandas as pd


class BaseNB(base.MiniBatchClassifier):
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
        pd = utils.pandas.import_pandas()
        jll = self.joint_log_likelihood_many(X)
        if jll.empty:
            return jll
        lse = pd.Series(special.logsumexp(jll, axis=1))
        return np.exp(jll.subtract(lse.values, axis="rows"))

    @property
    def _multiclass(self):
        return True

    def _unit_test_skips(self):
        # joint_log_likelihood_many's output is mis-aligned with the input batch
        # when the model has been trained via learn_one (rather than learn_many),
        # so predict_many/predict_proba_many disagree with their one-at-a-time
        # counterparts. Tracked separately.
        # `learn_many` also consumes sparse count matrices rather than a dense feature
        # frame, so the generic dense equivalence check does not apply; the learn_many vs
        # learn_one equivalence is covered by the dedicated naive Bayes test suite.
        return {
            "check_predict_many_matches_predict_one",
            "check_predict_proba_many_matches_predict_proba_one",
            "check_learn_many_matches_learn_one",
        }


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
    pd = utils.pandas.import_pandas()
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
    pd = utils.pandas.import_pandas()
    classes = np.unique(y)
    indices = np.searchsorted(classes, y)
    indptr = np.hstack((0, np.cumsum(np.isin(y, classes))))
    data = np.empty_like(indices)
    data.fill(1)
    return pd.DataFrame.sparse.from_spmatrix(
        sparse.csr_matrix((data, indices, indptr), shape=(y.shape[0], classes.shape[0])),
        index=y.index,
        columns=[str(c) for c in classes],
    )
