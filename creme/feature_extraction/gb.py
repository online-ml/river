import collections
import copy

from .. import base
from .. import stats


__all__ = ['GroupBy']


class GroupBy(base.Transformer):
    """

    Example
    -------

        #!python
        >>> import creme

        >>> X = [
        ...     {'place': 'Taco Bell', 'revenue': 42},
        ...     {'place': 'Burger King', 'revenue': 16},
        ...     {'place': 'Burger King', 'revenue': 24},
        ...     {'place': 'Taco Bell', 'revenue': 58},
        ...     {'place': 'Burger King', 'revenue': 20},
        ...     {'place': 'Taco Bell', 'revenue': 50}
        ... ]

        >>> agg = creme.feature_extraction.GroupBy(
        ...     on='revenue',
        ...     by='place',
        ...     how=creme.stats.Mean()
        ... )

        >>> for x in X:
        ...     print(agg.fit_one(x))
        {'revenue_mean_by_place': 42.0}
        {'revenue_mean_by_place': 16.0}
        {'revenue_mean_by_place': 20.0}
        {'revenue_mean_by_place': 50.0}
        {'revenue_mean_by_place': 20.0}
        {'revenue_mean_by_place': 50.0}

    """

    def __init__(self, on: str, by: str, how: stats.RunningStatistic):
        self.on = on
        self.by = by
        self.how = how
        self.stats = collections.defaultdict(lambda: copy.deepcopy(how))

    def fit_one(self, x, y=None):
        self.stats[x[self.by]].update(x[self.on])
        return self.transform_one(x)

    def transform_one(self, x):
        return {f'{self.on}_{self.how.name}_by_{self.by}': self.stats[x[self.by]].get()}
