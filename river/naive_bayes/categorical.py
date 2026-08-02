from __future__ import annotations

import collections
import math
import typing

from river import utils

from . import base

if typing.TYPE_CHECKING:
    import pandas as pd

__all__ = ["CategoricalNB"]


class CategoricalNB(base.BaseNB):
    """Naive Bayes classifier for categorical features.

    The input vector must contain categorical (discrete) feature values, for instance
    strings such as `{"weather": "sunny", "wind": "strong"}`. Each feature is assumed to
    follow a categorical distribution, conditioned on the class. This mirrors scikit-learn's
    [`CategoricalNB`](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.CategoricalNB.html),
    but learns incrementally: new feature values (categories) encountered after the first
    observations are handled gracefully.

    Parameters
    ----------
    alpha
        Additive (Laplace/Lidstone) smoothing parameter (use 0 for no smoothing).

    Attributes
    ----------
    class_counts : collections.Counter
        Number of times each class has been seen.
    feature_counts : collections.defaultdict
        Number of times each `(class, category)` pair has been seen, per feature.
    category_counts : collections.defaultdict
        Number of times each category has been seen, per feature. Used to count the
        number of distinct categories of a feature, which determines the smoothing
        denominator.

    Examples
    --------

    >>> from river import naive_bayes

    >>> dataset = [
    ...     ({"weather": "sunny", "humidity": "high"}, "no"),
    ...     ({"weather": "sunny", "humidity": "high"}, "no"),
    ...     ({"weather": "overcast", "humidity": "high"}, "yes"),
    ...     ({"weather": "rainy", "humidity": "normal"}, "yes"),
    ...     ({"weather": "rainy", "humidity": "normal"}, "yes"),
    ...     ({"weather": "overcast", "humidity": "normal"}, "yes"),
    ... ]

    >>> model = naive_bayes.CategoricalNB(alpha=1)

    >>> for x, y in dataset:
    ...     model.learn_one(x, y)

    >>> model.p_class("yes")
    0.666666...

    >>> model.predict_proba_one({"weather": "overcast", "humidity": "normal"})
    {'no': 0.08, 'yes': 0.92}

    >>> model.predict_one({"weather": "overcast", "humidity": "normal"})
    'yes'

    References
    ----------
    [^1]: [scikit-learn CategoricalNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.CategoricalNB.html)

    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.class_counts: collections.Counter = collections.Counter()
        self.feature_counts: collections.defaultdict = collections.defaultdict(collections.Counter)
        self.category_counts: collections.defaultdict = collections.defaultdict(collections.Counter)

    def learn_one(self, x, y):
        """Update the model with a single observation.

        Parameters
        ----------
        x
            Dictionary of categorical feature values.
        y
            Target class.

        """
        self.class_counts[y] += 1
        for f, value in x.items():
            self.feature_counts[f][(y, value)] += 1
            self.category_counts[f][value] += 1

    @property
    def classes_(self):
        return list(self.class_counts.keys())

    def p_class(self, c) -> float:
        return self.class_counts[c] / sum(self.class_counts.values())

    def p_feature_given_class(self, f, value, c) -> float:
        """Probability of a category given a class for a feature, with smoothing."""
        n_categories = len(self.category_counts.get(f, ())) or 1
        num = self.feature_counts.get(f, {}).get((c, value), 0.0) + self.alpha
        den = self.class_counts[c] + self.alpha * n_categories
        return num / den

    def joint_log_likelihood(self, x):
        """Compute the unnormalized posterior log-likelihood of `x`.

        The log-likelihood is `log P(c) + log P(x|c)`.

        """
        if not self.class_counts:
            return {}
        return {
            c: math.log(self.p_class(c))
            + sum(math.log(self.p_feature_given_class(f, value, c)) for f, value in x.items())
            for c in self.classes_
        }

    def learn_many(self, X: pd.DataFrame, y: pd.Series):
        """Learn from a batch of observations.

        Parameters
        ----------
        X
            A dataframe of categorical feature values.
        y
            A series of target classes.

        """
        for (_, row), label in zip(X.iterrows(), y):
            self.learn_one(row.to_dict(), label)

    def joint_log_likelihood_many(self, X: pd.DataFrame) -> pd.DataFrame:
        """Compute the unnormalized posterior log-likelihood of `X` in mini-batches."""
        pd = utils.pandas.import_pandas()
        index = X.index
        if not self.class_counts:
            return pd.DataFrame(index=index)
        records = [self.joint_log_likelihood(row.to_dict()) for _, row in X.iterrows()]
        return pd.DataFrame(records, index=index, columns=self.classes_)
