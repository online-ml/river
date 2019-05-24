import collections
import copy
import functools
import typing

from .. import base


__all__ = ['Agg', 'TargetAgg']


class Agg(base.Transformer):
    """Computes a streaming aggregate.

    At each step, the running statistic ``how`` of group ``by`` is updated with the value of
    ``on``. You can combine this with a ``creme.compose.TransformerUnion`` to extract many
    aggregate statistics in one go. The keys of the ``dict`` containing the aggregates is
    automatically guessed from ``how``, ``by``, and ``on``.

    Parameters:
        on (str): The feature on which to compute.
        by (str): The feature on which to group.
        how (stats.Univariate): The statistic to compute.

    Attributes:
        groups (collections.defaultdict): Maps group keys to univariate statistics.
        feature_name (str): The name of the feature in the output.

    Example:

        ::

            >>> from creme import feature_extraction
            >>> from creme import stats

            >>> X = [
            ...     {'place': 'Taco Bell', 'revenue': 42},
            ...     {'place': 'Burger King', 'revenue': 16},
            ...     {'place': 'Burger King', 'revenue': 24},
            ...     {'place': 'Taco Bell', 'revenue': 58},
            ...     {'place': 'Burger King', 'revenue': 20},
            ...     {'place': 'Taco Bell', 'revenue': 50}
            ... ]

            >>> agg = feature_extraction.Agg(
            ...     on='revenue',
            ...     by='place',
            ...     how=stats.Mean()
            ... )

            >>> for x in X:
            ...     print(agg.fit_one(x).transform_one(x))
            {'revenue_mean_by_place': 42.0}
            {'revenue_mean_by_place': 16.0}
            {'revenue_mean_by_place': 20.0}
            {'revenue_mean_by_place': 50.0}
            {'revenue_mean_by_place': 20.0}
            {'revenue_mean_by_place': 50.0}

    References:
        1. `Streaming groupbys in pandas for big datasets <(https://maxhalford.github.io/blog/streaming-groupbys-in-pandas-for-big-datasets/>`_

    """

    def __init__(self, on, by, how):
        self.on = on
        self.by = by if isinstance(by, (typing.List, typing.Tuple)) else [by]
        self.how = how
        self.groups = collections.defaultdict(functools.partial(copy.deepcopy, how))
        self.feature_name = f'{self.on}_{self.how.name}_by_{"_and_".join(self.by)}'

    def _get_key(self, x):
        return '_'.join(str(x[k]) for k in self.by)

    def fit_one(self, x, y=None):
        self.groups[self._get_key(x)].update(x[self.on])
        return self

    def transform_one(self, x):
        return {self.feature_name: self.groups[self._get_key(x)].get()}

    def __str__(self):
        return self.feature_name


class TargetAgg(base.Transformer):
    """Computes a streaming group by on the target.

    At each step, the running statistic ``how`` of group ``by`` is updated with the target.

    Parameters:
        by (str): The feature on which to group.
        how (stats.Univariate): The statistic to compute.
        target_name (str): Name used in the result.

    Attributes:
        groups (dict): Maps group keys to univariate statistics.
        feature_name (str): The name of the feature in the output.

    Example:

        ::

            >>> import creme

            >>> X = [
            ...     {'place': 'Taco Bell', 'revenue': 42},
            ...     {'place': 'Burger King', 'revenue': 16},
            ...     {'place': 'Burger King', 'revenue': 24},
            ...     {'place': 'Taco Bell', 'revenue': 58},
            ...     {'place': 'Burger King', 'revenue': 20},
            ...     {'place': 'Taco Bell', 'revenue': 50}
            ... ]

            >>> agg = creme.feature_extraction.TargetAgg(
            ...     by='place',
            ...     how=creme.stats.BayesianMean(
            ...         prior=3,
            ...         prior_weight=1
            ...     )
            ... )

            >>> for x in X:
            ...     print(agg.transform_one(x))
            ...     y = x.pop('revenue')
            ...     agg = agg.fit_one(x, y)
            {'target_bayes_mean_by_place': 3.0}
            {'target_bayes_mean_by_place': 3.0}
            {'target_bayes_mean_by_place': 9.5}
            {'target_bayes_mean_by_place': 22.5}
            {'target_bayes_mean_by_place': 14.333333...}
            {'target_bayes_mean_by_place': 34.333333...}

    References:
        1. `Streaming groupbys in pandas for big datasets <(https://maxhalford.github.io/blog/streaming-groupbys-in-pandas-for-big-datasets/>`_

    """

    def __init__(self, by, how, target_name='target'):
        self.by = by if isinstance(by, (typing.List, typing.Tuple)) else [by]
        self.how = how
        self.target_name = target_name
        self.groups = collections.defaultdict(functools.partial(copy.deepcopy, how))
        self.feature_name = f'{target_name}_{how.name}_by_{"_and_".join(self.by)}'

    def is_supervised(self):
        return True

    def _get_key(self, x):
        return '_'.join(str(x[k]) for k in self.by)

    def fit_one(self, x, y=None):
        self.groups[self._get_key(x)].update(y)
        return self

    def transform_one(self, x):
        return {self.feature_name: self.groups[self._get_key(x)].get()}

    def __str__(self):
        return self.feature_name
