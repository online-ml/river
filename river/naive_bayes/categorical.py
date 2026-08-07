from __future__ import annotations

import collections
import functools
import math
import typing

import narwhals as nw
import numpy as np

from river import proba
from river.utils.dataframe import into_frame, to_native_frame, into_series

from . import base

if typing.TYPE_CHECKING:
    from narwhals.stable.v2.typing import IntoDataFrame, IntoSeries

__all__ = ["CategoricalNB"]


class CategoricalNB(base.BaseNB):
    def __init__(self, alpha=1.0):
        """Categorical Naive Bayes.

        Categorical Naive Bayes is suitable for discrete-valued features where each
        feature takes one of a finite set of categories. The model estimates the
        conditional probability of each category given a class using frequency counts
        with Laplace smoothing.

        Parameters
        ----------
        alpha
            Additive (Laplace/Lidstone) smoothing parameter (use 0 for no smoothing).

        Attributes
        ----------
        class_counts : collections.Counter
            Number of training samples observed for each class.
        feature_counts : collections.defaultdict
            Counts of each `(class, category)` pair for every feature.
        feature_values : collections.defaultdict
            Set of distinct categories observed for each feature.

        Examples
        --------

        >>> from river import naive_bayes
        >>>
        >>> model = naive_bayes.CategoricalNB(alpha=1.0)
        >>>
        >>> model.learn_one(
        ...     {"color": "red", "shape": "round"},
        ...     "apple"
        ... )
        >>>
        >>> model.learn_one(
        ...     {"color": "yellow", "shape": "long"},
        ...     "banana"
        ... )
        >>>
        >>> model.predict_proba_one(
        ...     {"color": "red", "shape": "round"}
        ... )
        {'apple': ..., 'banana': ...}

        The model also supports mini-batch learning and prediction through
        `learn_many` and `predict_many`.

        References
        ----------
        [^1]: Christopher M. Bishop.
            *Pattern Recognition and Machine Learning*.
            Springer, 2006.
        """
        self.alpha = alpha
        self.class_counts = collections.Counter()
        self.feature_counts = collections.defaultdict(collections.Counter)
        self.feature_values = collections.defaultdict(set)

    def learn_one(self, x, y):
        """Update the model with a single observation.

        Parameters
        ----------
        x
            Dictionary mapping feature names to categorical values.
        y
            Target class label.
        """
        self.class_counts[y] += 1
        for f, value in x.items():
            self.feature_counts[f][(y, value)] += 1
            self.feature_values[f].add(value)

    def learn_many(self, X, y):
        """Update the model with a batch of observations.

        Parameters
        ----------
        X
            DataFrame containing categorical features.
        y
            Target class labels.

        Returns
        -------
        self
        """
        X = into_frame(X)
        y = into_series(y)

        class_counts = y.value_counts()

        for label, count in class_counts.iter_rows():
            self.class_counts[label] += count

        for f in X.columns:
            df = X.select(nw.col(f).alias("value"), y.alias("label"))

            counts = df.group_by(["label", "value"]).agg(nw.len().alias("count"))

            for label, value, count in counts.iter_rows():
                self.feature_counts[f][(label, value)] += count

            value_count = df.group_by("value").agg(nw.len().alias("count"))

            for row in value_count.iter_rows():
                self.feature_values[f].add(row[0])

    @property
    def classes_(self):
        return list(self.class_counts.keys())

    def p_class(self, c):
        return self.class_counts[c] / sum(self.class_counts.values())

    def p_feature_given_class(self, f, value, c):
        categories = len(self.feature_values[f]) or 1
        num = self.feature_counts[f][(c, value)] + self.alpha
        den = self.class_counts[c] + self.alpha * categories
        return num / den

    def joint_log_likelihood(self, x):
        """Compute the joint log-likelihood for a single observation.

        Parameters
        ----------
        x
            Dictionary mapping feature names to categorical values.

        Returns
        -------
        dict
            Mapping of class labels to unnormalized log posterior probabilities.
        """
        if not self.class_counts:
            return {}
        return {
            c: math.log(self.p_class(c))
            + sum(math.log(self.p_feature_given_class(f, value, c)) for f, value in x.items())
            for c in self.classes_
        }

    def joint_log_likelihood_many(self, X):
        X = into_frame(X)

        if not self.class_counts:
            native = X.to_native()
            return native.iloc[:, 0:0]

        jll = {}

        rows = [dict(zip(X.columns, row)) for row in X.iter_rows()]

        for c in self.classes_:
            ll = np.empty(len(rows), dtype=float)

            for i, row in enumerate(rows):
                ll[i] = math.log(self.p_class(c)) + sum(
                    math.log(self.p_feature_given_class(f, value, c)) for f, value in row.items()
                )

            jll[c] = ll

        return to_native_frame(jll, like=X)
