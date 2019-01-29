import collections
import copy

from .. import base
from .. import stats


__all__ = ['GroupBy']


class GroupBy(base.Transformer):
    """Computes a streaming group by.

    At each step, the running statistic ``how`` of group ``by`` is updated with the value of
    ``on``. You can combine this with a ``creme.compose.TransformerUnion`` to extract many
    aggregate statistics in one go. The keys of the ``dict`` containing the aggregates is
    automatically guessed from ``how``, ``by``, and ``on``.

    Parameters:
        on (str): The feature on which to compute.
        by (str): The feature on which to group.
        how (stats.RunningStatistic): The statistic to compute.

    Attributes:
        groups (dict): Maps grouping keys to instances of ``creme.stats.RunningStatistic``

    Example:

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

    References:

    - [Streaming groupbys in pandas for big datasets](https://maxhalford.github.io/blog/streaming-groupbys-in-pandas-for-big-datasets/)

    """

    def __init__(self, on: str, by: str, how: stats.RunningStatistic):
        self.on = on
        self.by = by
        self.how = how
        self.groups = collections.defaultdict(lambda: copy.deepcopy(how))

    def fit_one(self, x, y=None):
        self.groups[x[self.by]].update(x[self.on])
        return self.transform_one(x)

    def transform_one(self, x):
        return {f'{self.on}_{self.how.name}_by_{self.by}': self.groups[x[self.by]].get()}
