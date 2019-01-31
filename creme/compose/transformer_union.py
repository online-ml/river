import collections

from .. import base


__all__ = ['TransformerUnion']


class TransformerUnion(base.Transformer):
    """Groups multiple transformers into a single one.

    Calling ``transform_one`` will concatenate each transformer's output using a
    ``collections.ChainMap``.

    Args:
        transformers (list)

    Example:

    ::

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
        >>> agg = creme.compose.TransformerUnion([mean, count])

        >>> for x in X:
        ...     print(sorted(agg.fit_one(x).items()))
        [('revenue_count_by_place', 1), ('revenue_mean_by_place', 42.0)]
        [('revenue_count_by_place', 1), ('revenue_mean_by_place', 16.0)]
        [('revenue_count_by_place', 2), ('revenue_mean_by_place', 20.0)]
        [('revenue_count_by_place', 2), ('revenue_mean_by_place', 50.0)]
        [('revenue_count_by_place', 3), ('revenue_mean_by_place', 20.0)]
        [('revenue_count_by_place', 3), ('revenue_mean_by_place', 50.0)]

    """

    def __init__(self, transformers):
        self.transformers = transformers

    def fit_one(self, x, y=None):
        return dict(collections.ChainMap(*(t.fit_one(x, y) for t in self.transformers)))

    def transform_one(self, x):
        return dict(collections.ChainMap(*(t.transform_one(x) for t in self.transformers)))
