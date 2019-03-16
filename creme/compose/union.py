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
        >>> agg = creme.compose.TransformerUnion([
        ...     ('mean', mean),
        ...     ('count', count)
        ... ])

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

    def __init__(self, transformers):
        self.transformers = transformers

    def fit_one(self, x, y=None):
        for _, transformer in self.transformers:
            transformer.fit_one(x, y)
        return self

    def transform_one(self, x):
        return dict(collections.ChainMap(*(
            transformer.transform_one(x)
            for _, transformer in self.transformers
        )))
