import collections
import copy
import functools
import typing

from river import base
from river import stats


__all__ = ["Agg", "TargetAgg"]


class Agg(base.Transformer):
    """Computes a streaming aggregate.

    This transformer allows to compute an aggregate statistic, very much like the groupby method
    from `pandas`, but on a streaming dataset. This makes use of the streaming statistics from the
    `stats` module.

    When `learn_one` is called, the running statistic `how` of group `by` is updated with the value
    of `on`. Meanwhile, the output of `transform_one` is a single-element dictionary, where the key
    is the name of the aggregate and the value is the current value of the statistic for the
    relevant group. The key is automatically inferred from the parameters.

    Note that you can use a `compose.TransformerUnion` to extract many aggregate statistics in a
    concise manner.

    Parameters
    ----------
    on
        The feature on which to compute the aggregate statistic.
    by
        The feature by which to group the data.
    how
        The statistic to compute.

    Attributes
    ----------
    groups : collections.defaultdict
        Maps group keys to univariate statistics.
    feature_name : str
        The name of the feature used in the output.

    Examples
    --------

    Consider the following dataset:

    >>> X = [
    ...     {'country': 'France', 'place': 'Taco Bell', 'revenue': 42},
    ...     {'country': 'Sweden', 'place': 'Burger King', 'revenue': 16},
    ...     {'country': 'France', 'place': 'Burger King', 'revenue': 24},
    ...     {'country': 'Sweden', 'place': 'Taco Bell', 'revenue': 58},
    ...     {'country': 'Sweden', 'place': 'Burger King', 'revenue': 20},
    ...     {'country': 'France', 'place': 'Taco Bell', 'revenue': 50},
    ...     {'country': 'France', 'place': 'Burger King', 'revenue': 10},
    ...     {'country': 'Sweden', 'place': 'Taco Bell', 'revenue': 80}
    ... ]

    As an example, we can calculate the average (how) revenue (on) for each place (by):

    >>> from river import feature_extraction as fx
    >>> from river import stats

    >>> agg = fx.Agg(
    ...     on='revenue',
    ...     by='place',
    ...     how=stats.Mean()
    ... )

    >>> for x in X:
    ...     agg = agg.learn_one(x)
    ...     print(agg.transform_one(x))
    {'revenue_mean_by_place': 42.0}
    {'revenue_mean_by_place': 16.0}
    {'revenue_mean_by_place': 20.0}
    {'revenue_mean_by_place': 50.0}
    {'revenue_mean_by_place': 20.0}
    {'revenue_mean_by_place': 50.0}
    {'revenue_mean_by_place': 17.5}
    {'revenue_mean_by_place': 57.5}

    You can compute an aggregate over multiple keys by passing a tuple to the `by` argument.
    For instance, we can compute the maximum (how) revenue (on) per place as well as per
    day (by):

    >>> agg = fx.Agg(
    ...     on='revenue',
    ...     by=['place', 'country'],
    ...     how=stats.Max()
    ... )

    >>> for x in X:
    ...     agg = agg.learn_one(x)
    ...     print(agg.transform_one(x))
    {'revenue_max_by_place_and_country': 42}
    {'revenue_max_by_place_and_country': 16}
    {'revenue_max_by_place_and_country': 24}
    {'revenue_max_by_place_and_country': 58}
    {'revenue_max_by_place_and_country': 20}
    {'revenue_max_by_place_and_country': 50}
    {'revenue_max_by_place_and_country': 24}
    {'revenue_max_by_place_and_country': 80}

    You can use a `compose.TransformerUnion` in order to calculate multiple aggregates in one
    go. The latter can be constructed by using the `+` operator:

    >>> agg = (
    ...     fx.Agg(on='revenue', by='place', how=stats.Mean()) +
    ...     fx.Agg(on='revenue', by=['place', 'country'], how=stats.Max())
    ... )

    >>> import pprint
    >>> for x in X:
    ...     agg = agg.learn_one(x)
    ...     pprint.pprint(agg.transform_one(x))
    {'revenue_max_by_place_and_country': 42, 'revenue_mean_by_place': 42.0}
    {'revenue_max_by_place_and_country': 16, 'revenue_mean_by_place': 16.0}
    {'revenue_max_by_place_and_country': 24, 'revenue_mean_by_place': 20.0}
    {'revenue_max_by_place_and_country': 58, 'revenue_mean_by_place': 50.0}
    {'revenue_max_by_place_and_country': 20, 'revenue_mean_by_place': 20.0}
    {'revenue_max_by_place_and_country': 50, 'revenue_mean_by_place': 50.0}
    {'revenue_max_by_place_and_country': 24, 'revenue_mean_by_place': 17.5}
    {'revenue_max_by_place_and_country': 80, 'revenue_mean_by_place': 57.5}

    References
    ----------
    [^1]: [Streaming groupbys in pandas for big datasets](https://maxhalford.github.io/blog/streaming-groupbys-in-pandas-for-big-datasets/)

    """

    def __init__(self, on: str, by: typing.Union[str, typing.List[str]], how: stats.Univariate):
        self.on = on
        self.by = by if isinstance(by, list) else [by]
        self.how = how
        self.groups = collections.defaultdict(functools.partial(copy.deepcopy, how))
        self.feature_name = f'{self.on}_{self.how.name}_by_{"_and_".join(self.by)}'

    def _get_key(self, x):
        return "_".join(str(x[k]) for k in self.by)

    def learn_one(self, x):
        self.groups[self._get_key(x)].update(x[self.on])
        return self

    def transform_one(self, x):
        return {self.feature_name: self.groups[self._get_key(x)].get()}

    def __str__(self):
        return self.feature_name


class TargetAgg(base.SupervisedTransformer):
    """Computes a streaming aggregate of the target values.

    This transformer is identical to `feature_extraction.Agg`, the only difference is that it
    operates on the target rather than on a feature. At each step, the running statistic `how` of
    target values in group `by` is updated with the target. It is therefore a supervised
    transformer.

    Parameters
    ----------
    by
        The feature by which to group the target values.
    how
        The statistic to compute.
    target_name
        The target name which is used in the result.

    Attributes
    ----------
    groups
        Maps group keys to univariate statistics.
    feature_name
        The name of the feature in the output.

    Examples
    --------

    Consider the following dataset, where the second value of each value is the target:

    >>> dataset = [
    ...     ({'country': 'France', 'place': 'Taco Bell'}, 42),
    ...     ({'country': 'Sweden', 'place': 'Burger King'}, 16),
    ...     ({'country': 'France', 'place': 'Burger King'}, 24),
    ...     ({'country': 'Sweden', 'place': 'Taco Bell'}, 58),
    ...     ({'country': 'Sweden', 'place': 'Burger King'}, 20),
    ...     ({'country': 'France', 'place': 'Taco Bell'}, 50),
    ...     ({'country': 'France', 'place': 'Burger King'}, 10),
    ...     ({'country': 'Sweden', 'place': 'Taco Bell'}, 80)
    ... ]

    As an example, let's perform a target encoding of the `place` feature. Instead of simply
    updating a running average, we use a `stats.BayesianMean` which allows us to incorporate
    some prior knowledge. This makes subsequent models less prone to overfitting. Indeed, it
    dampens the fact that too few samples might have been seen within a group.

    >>> from river import feature_extraction
    >>> from river import stats

    >>> agg = feature_extraction.TargetAgg(
    ...     by='place',
    ...     how=stats.BayesianMean(
    ...         prior=3,
    ...         prior_weight=1
    ...     )
    ... )

    >>> for x, y in dataset:
    ...     print(agg.transform_one(x))
    ...     agg = agg.learn_one(x, y)
    {'target_bayes_mean_by_place': 3.0}
    {'target_bayes_mean_by_place': 3.0}
    {'target_bayes_mean_by_place': 9.5}
    {'target_bayes_mean_by_place': 22.5}
    {'target_bayes_mean_by_place': 14.333}
    {'target_bayes_mean_by_place': 34.333}
    {'target_bayes_mean_by_place': 15.75}
    {'target_bayes_mean_by_place': 38.25}

    Just like with `feature_extraction.Agg`, we can specify multiple features on which to
    group the data:

    >>> agg = feature_extraction.TargetAgg(
    ...     by=['place', 'country'],
    ...     how=stats.BayesianMean(
    ...         prior=3,
    ...         prior_weight=1
    ...     )
    ... )

    >>> for x, y in dataset:
    ...     print(agg.transform_one(x))
    ...     agg = agg.learn_one(x, y)
    {'target_bayes_mean_by_place_and_country': 3.0}
    {'target_bayes_mean_by_place_and_country': 3.0}
    {'target_bayes_mean_by_place_and_country': 3.0}
    {'target_bayes_mean_by_place_and_country': 3.0}
    {'target_bayes_mean_by_place_and_country': 9.5}
    {'target_bayes_mean_by_place_and_country': 22.5}
    {'target_bayes_mean_by_place_and_country': 13.5}
    {'target_bayes_mean_by_place_and_country': 30.5}

    References
    ----------
    1. [Streaming groupbys in pandas for big datasets](https://maxhalford.github.io/blog/streaming-groupbys-in-pandas-for-big-datasets/)

    """

    def __init__(
        self,
        by: typing.Union[str, typing.List[str]],
        how: stats.Univariate,
        target_name="target",
    ):
        self.by = by if isinstance(by, list) else [by]
        self.how = how
        self.target_name = target_name
        self.groups = collections.defaultdict(functools.partial(copy.deepcopy, how))
        self.feature_name = f'{target_name}_{how.name}_by_{"_and_".join(self.by)}'

    def _get_key(self, x):
        return "_".join(str(x[k]) for k in self.by)

    def learn_one(self, x, y):
        self.groups[self._get_key(x)].update(y)
        return self

    def transform_one(self, x):
        return {self.feature_name: self.groups[self._get_key(x)].get()}

    def __str__(self):
        return self.feature_name
