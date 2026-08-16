from __future__ import annotations

import collections
import math
import typing

import narwhals as nw
import numpy as np
import pandas as pd

from river.utils.dataframe import into_frame, to_native_frame

from . import base

if typing.TYPE_CHECKING:
    from narwhals.stable.v2.typing import IntoDataFrame, IntoSeries

__all__ = ["CategoricalNB"]


def _feature_counts():
    return collections.defaultdict(collections.Counter)


class CategoricalNB(base.BaseNB):
    """Naive Bayes classifier for categorical features.

    Categorical Naive Bayes learns a separate model for each categorical feature. For each class,
    a frequency is maintained for every value seen for every feature. Prediction is done by
    computing the joint log-likelihood of each class given the observed category values.

    Parameters
    ----------
    alpha
        Additive (Laplace/Lidstone) smoothing parameter (use 0 for no smoothing).

    Attributes
    ----------
    class_counts : collections.Counter
        Number of times each class has been seen.
    feature_counts : collections.defaultdict
        Total frequencies per feature, value, and class.

    Examples
    --------

    >>> from river import naive_bayes

    >>> X = [
    ...     {"outlook": "sunny", "temp": "hot"},
    ...     {"outlook": "sunny", "temp": "hot"},
    ...     {"outlook": "sunny", "temp": "mild"},
    ...     {"outlook": "sunny", "temp": "cool"},
    ...     {"outlook": "rainy", "temp": "cool"},
    ...     {"outlook": "rainy", "temp": "cool"},
    ...     {"outlook": "rainy", "temp": "mild"},
    ...     {"outlook": "rainy", "temp": "hot"},
    ...     {"outlook": "overcast", "temp": "cool"},
    ...     {"outlook": "overcast", "temp": "mild"},
    ...     {"outlook": "overcast", "temp": "mild"},
    ...     {"outlook": "overcast", "temp": "hot"},
    ... ]

    >>> y = ["no", "no", "no", "no", "yes", "yes", "yes", "yes",
    ...      "yes", "yes", "yes", "yes"]

    >>> model = naive_bayes.CategoricalNB(alpha=1)

    >>> for x, yi in zip(X, y):
    ...     model.learn_one(x, yi)

    >>> model.predict_one({"outlook": "sunny", "temp": "mild"})
    'no'

    >>> model.predict_proba_one({"outlook": "sunny", "temp": "mild"})
    {'no': 0.755..., 'yes': 0.244...}

    You can also train and predict in mini-batch mode.

    >>> import pandas as pd

    >>> df = pd.DataFrame(X)
    >>> y = pd.Series(y)

    >>> batch_model = naive_bayes.CategoricalNB(alpha=1)
    >>> batch_model.learn_many(df, y)

    >>> unseen = pd.DataFrame([{"outlook": "rainy", "temp": "cool"}])

    >>> batch_model.predict_many(unseen)
    0    yes
    dtype: object

    >>> batch_model.predict_proba_many(unseen)
               no       yes
    0  0.109900  0.890100

    References
    ----------
    [^1]: [CategoricalNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.CategoricalNB.html)

    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.class_counts = collections.Counter()
        self.feature_counts = collections.defaultdict(_feature_counts)

    def learn_one(self, x, y):
        """Updates the model with a single observation.

        Parameters
        ----------
        x
            Dictionary of categorical features.
        y
            Target class.

        """
        self.class_counts.update((y,))

        for f, v in x.items():
            self.feature_counts[f][v].update({y: 1})

    def p_feature_given_class(self, f: str, v, c: str) -> float:
        num = self.feature_counts.get(f, {}).get(v, {}).get(c, 0.0) + self.alpha
        feature_total = sum(
            values.get(c, 0.0) for values in self.feature_counts.get(f, {}).values()
        )
        n_categories = max(1, len(self.feature_counts[f]))
        den = feature_total + self.alpha * n_categories
        return num / den

    def p_class(self, c: str) -> float:
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
            Dictionary of categorical features.

        Returns
        -------
        Mapping between classes and joint log likelihood.

        """
        return {
            c: math.log(self.p_class(c))
            + sum(math.log(self.p_feature_given_class(f, v, c)) for f, v in x.items())
            for c in self.class_counts
        }

    def learn_many(self, X: IntoDataFrame, y: IntoSeries):
        """Learn from a batch of categorical feature vectors.

        Parameters
        ----------
        X
            Feature vectors, one categorical column per feature.
        y
            Target classes.

        """
        if hasattr(X, "sparse"):
            X = X.sparse.to_dense()

        X = nw.from_native(X, eager_only=True)
        y = nw.from_native(y, series_only=True)

        self.class_counts.update(y.to_list())

        X_np = np.asarray(X.to_numpy(), dtype=object)
        y_np = np.asarray(y.to_numpy())

        for j, col in enumerate(X.columns):
            values = X_np[:, j]
            for c in np.unique(y_np):
                mask = y_np == c
                col_values = values[mask]
                col_values = col_values[pd.notna(col_values)]

                for v in col_values:
                    self.feature_counts[col][v].update({c: 1})

    def joint_log_likelihood_many(self, X: IntoDataFrame) -> IntoDataFrame:
        """Computes the joint log likelihood of input features.

        Parameters
        ----------
        X
            Feature vectors, one categorical column per feature.

        Returns
        -------
        Input samples joint log likelihood.

        """
        if hasattr(X, "sparse"):
            X = X.sparse.to_dense()

        X = nw.from_native(X, eager_only=True)
        X = into_frame(X)

        if not self.class_counts:
            native = X.to_native()
            return native.iloc[:, 0:0]

        jll = {}

        for c in self.class_counts:
            ll = np.full(len(X), math.log(self.p_class(c)), dtype=float)

            for col in X.columns:
                values = np.asarray(
                    X.select(nw.col(col)).to_numpy(),
                    dtype=object,
                ).ravel()

                for i, v in enumerate(values):
                    if pd.notna(v):
                        ll[i] += math.log(self.p_feature_given_class(col, v, c))

            jll[c] = ll
        return to_native_frame(jll, like=X)
