import collections

from .. import base

from . import func


__all__ = ['TransformerUnion']


class TransformerUnion(collections.UserDict, base.Transformer):
    """Packs multiple transformers into a single one.

    Calling ``transform_one`` will concatenate each transformer's output using a
    ``collections.ChainMap``.

    Args:
        transformers (list)

    Example:

    ::

        >>> from pprint import pprint
        >>> import creme.compose
        >>> import creme.feature_extraction

        >>> X = [
        ...     {'place': 'Taco Bell', 'revenue': 42},
        ...     {'place': 'Burger King', 'revenue': 16},
        ...     {'place': 'Burger King', 'revenue': 24},
        ...     {'place': 'Taco Bell', 'revenue': 58},
        ...     {'place': 'Burger King', 'revenue': 20},
        ...     {'place': 'Taco Bell', 'revenue': 50}
        ... ]

        >>> mean = creme.feature_extraction.GroupBy(
        ...     on='revenue',
        ...     by='place',
        ...     how=creme.stats.Mean()
        ... )
        >>> count = creme.feature_extraction.GroupBy(
        ...     on='revenue',
        ...     by='place',
        ...     how=creme.stats.Count()
        ... )
        >>> agg = creme.compose.TransformerUnion([mean])
        >>> agg += count

        >>> for x in X:
        ...     pprint(agg.fit_one(x).transform_one(x))
        {'revenue_count_by_place': 1, 'revenue_mean_by_place': 42.0}
        {'revenue_count_by_place': 1, 'revenue_mean_by_place': 16.0}
        {'revenue_count_by_place': 2, 'revenue_mean_by_place': 20.0}
        {'revenue_count_by_place': 2, 'revenue_mean_by_place': 50.0}
        {'revenue_count_by_place': 3, 'revenue_mean_by_place': 20.0}
        {'revenue_count_by_place': 3, 'revenue_mean_by_place': 50.0}

        >>> pprint(agg.transform_one({'place': 'Taco Bell'}))
        {'revenue_count_by_place': 3, 'revenue_mean_by_place': 50.0}

    """

    def __init__(self, estimators=None):
        super().__init__()
        if estimators is not None:
            for estimator in estimators:
                self += estimator

    def __str__(self):
        """Return a human friendly representation of the pipeline."""
        return f'{{{", ".join(self.keys())}}}'

    def __add__(self, other):
        """Adds an estimator while taking care of the input type."""

        # Infer a name if none is given
        if not isinstance(other, (list, tuple)):
            other = (str(other), other)

        # If a function is given then wrap it in a FuncTransformer
        if callable(other[1]):
            other = (other[1].__name__, func.FuncTransformer(other[1]))

        # Prefer clarity to magic
        if other[0] in self:
            raise KeyError(f'{other[0]} already exists')

        # Store the estimator
        self[other[0]] = other[1]

        return self

    def fit_one(self, x, y=None):
        for estimator in self.values():
            estimator.fit_one(x, y)
        return self

    def transform_one(self, x):
        """Passes the data through each transformer and packs the results together."""
        return dict(collections.ChainMap(*(
            estimator.transform_one(x)
            for estimator in self.values()
        )))

    def fit_transform_one(self, x: dict, y=None):
        """Fits and transforms while ensuring no leakage occurs if the transformer is supervised."""
        return dict(collections.ChainMap(*(
            estimator.fit_transform_one(x, y)
            for estimator in self.values()
        )))
