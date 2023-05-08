from __future__ import annotations

import collections
import math

import numpy as np
import pandas as pd
from scipy import sparse

from river.base import tags

from . import base

__all__ = ["MultinomialNB"]


class MultinomialNB(base.BaseNB):
    """Naive Bayes classifier for multinomial models.

    Multinomial Naive Bayes model learns from occurrences between features such as word counts
    and discrete classes. The input vector must contain positive values, such as
    counts or TF-IDF values.


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
    ...     ("nb", naive_bayes.MultinomialNB(alpha=1))
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
    {'yes': 0.413, 'maybe': 0.310, 'no': 0.275}

    >>> model.predict_one("test")
    'yes'

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
    ...     ("nb", naive_bayes.MultinomialNB(alpha=1))
    ... )

    >>> model = model.learn_many(X, y)

    >>> unseen = pd.Series(["Taiwanese Taipei", "Chinese Shanghai"])

    >>> model.predict_proba_many(unseen)
          maybe        no       yes
    0  0.373272  0.294931  0.331797
    1  0.160396  0.126733  0.712871

    >>> model.predict_many(unseen)
    0    maybe
    1      yes
    dtype: object

    References
    ----------
    [^1]: [Naive Bayes text classification](https://nlp.stanford.edu/IR-book/html/htmledition/naive-bayes-text-classification-1.html)

    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.class_counts = collections.Counter()
        self.feature_counts = collections.defaultdict(collections.Counter)
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
            self.class_totals.update({y: frequency})

        return self

    @property
    def classes_(self):
        return list(self.class_counts.keys())

    @property
    def n_terms(self):
        return len(self.feature_counts)

    def p_feature_given_class(self, f, c):
        num = self.feature_counts.get(f, {}).get(c, 0.0) + self.alpha
        den = self.class_totals[c] + self.alpha * self.n_terms
        return num / den

    def p_class(self, c) -> float:
        return self.class_counts[c] / sum(self.class_counts.values())

    def p_class_many(self) -> pd.DataFrame:
        return base.from_dict(self.class_counts).T[list(self.class_counts.keys())] / sum(
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
        return {
            c: math.log(self.p_class(c))
            + sum(
                frequency * math.log(self.p_feature_given_class(f, c)) for f, frequency in x.items()
            )
            for c in self.classes_
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

    def _feature_log_prob(self, columns: list, known: list, unknown: list) -> pd.DataFrame:
        """Compute log probabilities of input features.

        Parameters
        ----------
        columns
            List of input features.
        known
            List of input features that are part of the vocabulary.
        unknown
            List of input features that are not part the vocabulary.

        Returns
        -------
        Log probabilities of input features.

        """
        smooth_fc = np.log(base.from_dict(self.feature_counts).fillna(0).T[known] + self.alpha)
        smooth_fc[unknown] = np.log(self.alpha)

        smooth_cc = np.log(base.from_dict(self.class_totals) + self.alpha * self.n_terms)

        return smooth_fc.subtract(smooth_cc.values, axis="rows")[columns].T

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
        known, unknown = [], []

        if not self.class_counts or not self.feature_counts:
            return pd.DataFrame(index=index)

        for f in columns:
            if f in self.feature_counts:
                known.append(f)
            else:
                unknown.append(f)

        if hasattr(X, "sparse"):
            X = sparse.csr_matrix(X.sparse.to_coo())

        return pd.DataFrame(
            X @ self._feature_log_prob(columns=columns, known=known, unknown=unknown)
            + np.log(self.p_class_many()).values,
            index=index,
            columns=self.class_totals.keys(),
        )
