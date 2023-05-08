from __future__ import annotations

import collections
import math

import numpy as np
import pandas as pd
from scipy import sparse

from river.base import tags

from . import base

__all__ = ["ComplementNB"]


class ComplementNB(base.BaseNB):
    """Naive Bayes classifier for multinomial models.

    Complement Naive Bayes model learns from occurrences between features such as word counts
    and discrete classes. ComplementNB is suitable for imbalance dataset.
    The input vector must contain positive values, such as counts or TF-IDF values.

    Parameters
    ----------
    alpha
        Additive (Laplace/Lidstone) smoothing parameter (use 0 for no smoothing).

    Attributes
    ----------
    class_dist : proba.Multinomial
        Class prior probability distribution.
    feature_counts : collections.defaultdict
        Total frequencies per feature and class.
    class_totals : collections.Counter
        Total frequencies per class.

    Examples
    --------

    >>> import pandas as pd
    >>> from river import compose
    >>> from river import feature_extraction
    >>> from river import naive_bayes

    >>> docs = [
    ...     ("Chinese Beijing Chinese", "yes"),
    ...     ("Chinese Chinese Shanghai", "yes"),
    ...     ("Chinese Macao", "maybe"),
    ...     ("Tokyo Japan Chinese", "no")
    ... ]

    >>> model = compose.Pipeline(
    ...     ("tokenize", feature_extraction.BagOfWords(lowercase=False)),
    ...     ("nb", naive_bayes.ComplementNB(alpha=1))
    ... )

    >>> for sentence, label in docs:
    ...     model = model.learn_one(sentence, label)

    >>> model["nb"].p_class("yes")
    0.5

    >>> model["nb"].p_class("no")
    0.25

    >>> model["nb"].p_class("maybe")
    0.25

    >>> model.predict_proba_one("test")
    {'yes': 0.275, 'maybe': 0.375, 'no': 0.35}

    >>> model.predict_one("test")
    'maybe'

    You can train the model and make predictions in mini-batch mode using the class methods
    `learn_many` and `predict_many`.

    >>> df_docs = pd.DataFrame(docs, columns = ["docs", "y"])

    >>> X = pd.Series([
    ...    "Chinese Beijing Chinese",
    ...    "Chinese Chinese Shanghai",
    ...    "Chinese Macao",
    ...    "Tokyo Japan Chinese"
    ... ])

    >>> y = pd.Series(["yes", "yes", "maybe", "no"])

    >>> model = compose.Pipeline(
    ...     ("tokenize", feature_extraction.BagOfWords(lowercase=False)),
    ...     ("nb", naive_bayes.ComplementNB(alpha=1))
    ... )

    >>> model = model.learn_many(X, y)

    >>> unseen = pd.Series(["Taiwanese Taipei", "Chinese Shanghai"])

    >>> model.predict_proba_many(unseen)
          maybe        no       yes
    0  0.415129  0.361624  0.223247
    1  0.248619  0.216575  0.534807

    >>> model.predict_many(unseen)
    0    maybe
    1      yes
    dtype: object

    References
    ----------
    [^1]: [Rennie, J.D., Shih, L., Teevan, J. and Karger, D.R., 2003. Tackling the poor assumptions of naive bayes text classifiers. In Proceedings of the 20th international conference on machine learning (ICML-03) (pp. 616-623)](https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf)
    [^2]: [StackExchange discussion](https://stats.stackexchange.com/questions/126009/complement-naive-bayes)

    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.class_counts = collections.Counter()
        self.feature_counts = collections.defaultdict(collections.Counter)
        self.feature_totals = collections.Counter()
        self.class_totals = collections.Counter()

    def _more_tags(self):
        return {tags.POSITIVE_INPUT}

    def learn_one(self, x, y):
        """Updates the model with a single observation.

        Parameters
        ----------
        x
            Dictionary of term frequencies.
        y
            Target class.

        Returns
        -------
        self

        """
        self.class_counts.update((y,))

        for f, frequency in x.items():
            self.feature_counts[f].update({y: frequency})
            self.feature_totals.update({f: frequency})
            self.class_totals.update({y: frequency})

        return self

    def p_class(self, c):
        return self.class_counts[c] / sum(self.class_counts.values())

    def p_class_many(self) -> pd.DataFrame:
        return base.from_dict(self.class_counts).T[self.class_counts] / sum(
            self.class_counts.values()
        )

    def joint_log_likelihood(self, x):
        """Computes the joint log likelihood of input features.

        Parameters
        ----------
        x
            Dictionary of term frequencies.

        Returns
        -------
        Mapping between classes and joint log likelihood.

        """
        cc = {
            c: {
                f: self.feature_totals[f] + self.alpha - frequency.get(c, 0)
                for f, frequency in self.feature_counts.items()
            }
            for c in self.class_counts
        }

        return {
            c: sum(
                {
                    f: frequency * -math.log(cc[c].get(f, self.alpha) / sum(cc[c].values()))
                    for f, frequency in x.items()
                }.values()
            )
            for c in self.class_counts
        }

    def learn_many(self, X: pd.DataFrame, y: pd.Series):
        """Learn from a batch of count vectors.

        Parameters
        ----------
        X
            Count vectors.
        y
            Target classes.

        Returns
        -------
        self

        """
        y = base.one_hot_encode(y)
        columns, classes = X.columns, y.columns
        y = sparse.csc_matrix(y.sparse.to_coo()).T

        self.class_counts.update({c: count.item() for c, count in zip(classes, y.sum(axis=1))})

        if hasattr(X, "sparse"):
            X = sparse.csr_matrix(X.sparse.to_coo())

        fc = y @ X

        self.class_totals.update({c: count.item() for c, count in zip(classes, fc.sum(axis=1))})

        self.feature_totals.update(
            {c: count.item() for c, count in zip(columns, np.array(fc.sum(axis=0)).flatten())}
        )

        # Update feature counts by slicing the sparse matrix per column.
        # Each column correspond to a class.
        for c, i in zip(classes, range(fc.shape[0])):
            counts = {c: {columns[f]: count for f, count in zip(fc[i].indices, fc[i].data)}}

            # Transform {classe_i: {token_1: f_1, ... token_n: f_n}} into:
            # [{token_1: {classe_i: f_1}},.. {token_n: {class_i: f_n}}]
            for dict_count in [
                {token: {c: f} for token, f in frequencies.items()}
                for c, frequencies in counts.items()
            ]:
                for f, count in dict_count.items():
                    self.feature_counts[f].update(count)

        return self

    def _feature_log_prob(self, unknown: list, columns: list) -> pd.DataFrame:
        """Compute log probabilities of input features.

        Parameters
        ----------
        unknown
            List of features that are not part the vocabulary.
        columns
            List of input features.

        Returns
        -------
        Log probabilities of input features.

        """
        cc = (
            base.from_dict(self.feature_totals).squeeze().T
            + self.alpha
            - base.from_dict(self.feature_counts).fillna(0).T
        )

        sum_cc = cc.sum(axis=1).values

        cc[unknown] = self.alpha

        return -np.log(cc[columns].T / sum_cc)

    def joint_log_likelihood_many(self, X: pd.DataFrame) -> pd.DataFrame:
        """Computes the joint log likelihood of input features.

        Parameters
        ----------
        X
            Term-frequency or TF-IDF pandas dataframe.

        Returns
        -------
        Input samples joint log likelihood.

        """
        index, columns = X.index, X.columns
        unknown = [x for x in columns if x not in self.feature_counts]

        if not self.class_counts or not self.feature_counts:
            return pd.DataFrame(index=index)

        if hasattr(X, "sparse"):
            X = sparse.csr_matrix(X.sparse.to_coo())

        return pd.DataFrame(
            X @ self._feature_log_prob(unknown=unknown, columns=columns),
            index=index,
            columns=self.class_counts.keys(),
        )
