from __future__ import annotations

import itertools

from river import base, utils

__all__ = ["PolynomialExtender"]


def powerset(iterable, min_size, max_size, with_replacement=False):
    """powerset([A, B, C], 1, 2) --> (A,) (B,) (C,) (A, B) (A, C) (B, C)"""
    combiner = (
        itertools.combinations_with_replacement if with_replacement else itertools.combinations
    )
    sizes = range(min_size, max_size + 1)
    return itertools.chain.from_iterable(combiner(list(iterable), size) for size in sizes)


class PolynomialExtender(base.Transformer):
    """Polynomial feature extender.

    Generate features consisting of all polynomial combinations of the features with degree less
    than or equal to the specified degree.

    Be aware that the number of outputted features scales polynomially in the number of input
    features and exponentially in the degree. High degrees can cause overfitting.

    Parameters
    ----------
    degree
        The maximum degree of the polynomial features.
    interaction_only
        If `True` then only combinations that include an element at most once will be computed.
    include_bias
        Whether or not to include a dummy feature which is always equal to 1.
    bias_name
        Name to give to the bias feature.

    Examples
    --------

    >>> from river import feature_extraction as fx

    >>> X = [
    ...     {'x': 0, 'y': 1},
    ...     {'x': 2, 'y': 3},
    ...     {'x': 4, 'y': 5}
    ... ]

    >>> poly = fx.PolynomialExtender(degree=2, include_bias=True)
    >>> for x in X:
    ...     print(poly.transform_one(x))
    {'x': 0, 'y': 1, 'x*x': 0, 'x*y': 0, 'y*y': 1, 'bias': 1}
    {'x': 2, 'y': 3, 'x*x': 4, 'x*y': 6, 'y*y': 9, 'bias': 1}
    {'x': 4, 'y': 5, 'x*x': 16, 'x*y': 20, 'y*y': 25, 'bias': 1}

    >>> X = [
    ...     {'x': 0, 'y': 1, 'z': 2},
    ...     {'x': 2, 'y': 3, 'z': 2},
    ...     {'x': 4, 'y': 5, 'z': 2}
    ... ]

    >>> poly = fx.PolynomialExtender(degree=3, interaction_only=True)
    >>> for x in X:
    ...     print(poly.transform_one(x))
    {'x': 0, 'y': 1, 'z': 2, 'x*y': 0, 'x*z': 0, 'y*z': 2, 'x*y*z': 0}
    {'x': 2, 'y': 3, 'z': 2, 'x*y': 6, 'x*z': 4, 'y*z': 6, 'x*y*z': 12}
    {'x': 4, 'y': 5, 'z': 2, 'x*y': 20, 'x*z': 8, 'y*z': 10, 'x*y*z': 40}

    Polynomial features are typically used for a linear model to capture interactions between
    features. This may done by setting up a pipeline, as so:

    >>> from river import datasets
    >>> from river import evaluate
    >>> from river import linear_model as lm
    >>> from river import metrics
    >>> from river import preprocessing as pp

    >>> dataset = datasets.Phishing()

    >>> model = (
    ...     fx.PolynomialExtender() |
    ...     pp.StandardScaler() |
    ...     lm.LogisticRegression()
    ... )

    >>> metric = metrics.Accuracy()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    Accuracy: 88.88%

    """

    def __init__(self, degree=2, interaction_only=False, include_bias=False, bias_name="bias"):
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self.bias_name = bias_name

    def _enumerate(self, keys):
        return powerset(
            keys,
            min_size=1,
            max_size=self.degree,
            with_replacement=not self.interaction_only,
        )

    def transform_one(self, x):
        features = {
            "*".join(map(str, sorted(combo))): utils.math.prod(x[c] for c in combo)
            for combo in self._enumerate(x.keys())
        }
        if self.include_bias:
            features[self.bias_name] = 1
        return features
