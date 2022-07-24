import collections
import copy
import functools
import typing

import pandas as pd

from river import base, stats


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
        The feature by which to group the data. All the data is included in the aggregate
        if this is `None`.
    how
        The statistic to compute.

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
    [^1]: [Streaming groupbys in pandas for big datasets](https://maxhalford.github.io/blog/pandas-streaming-groupby/)

    """

    def __init__(
        self,
        on: str,
        by: typing.Optional[typing.Union[str, typing.List[str]]],
        how: stats.base.Univariate,
    ):
        self.on = on
        self.by = (by if isinstance(by, list) else [by]) if by is not None else by
        self.how = how
        self._groups: typing.DefaultDict = collections.defaultdict(
            functools.partial(copy.deepcopy, how)
        )
        self._feature_name = f"{self.on}_{self.how.name}"
        if self.by:
            self._feature_name += f"_by_{'_and_'.join(self.by)}"

    def _make_key(self, x):
        if self.by:
            return tuple(x[k] for k in self.by)
        return None

    def learn_one(self, x):
        self._groups[self._make_key(x)].update(x[self.on])
        return self

    def transform_one(self, x):
        return {self._feature_name: self._groups[self._make_key(x)].get()}

    @property
    def state(self) -> pd.Series:
        """Return the current values for each group.

        Examples
        --------

        >>> X = [
        ...     {"country": "France", "place": "Taco Bell", "revenue": 42},
        ...     {"country": "Sweden", "place": "Burger King", "revenue": 16},
        ...     {"country": "France", "place": "Burger King", "revenue": 24},
        ...     {"country": "Sweden", "place": "Taco Bell", "revenue": 58},
        ...     {"country": "Sweden", "place": "Burger King", "revenue": 20},
        ...     {"country": "France", "place": "Taco Bell", "revenue": 50},
        ...     {"country": "France", "place": "Burger King", "revenue": 10},
        ...     {"country": "Sweden", "place": "Taco Bell", "revenue": 80},
        ... ]

        >>> from river import feature_extraction as fx
        >>> from river import stats

        >>> agg = fx.Agg(on="revenue", by="place", how=stats.Mean())
        >>> for x in X:
        ...     agg = agg.learn_one(x)
        >>> agg.state
        Taco Bell      57.5
        Burger King    17.5
        Name: revenue_mean_by_place, dtype: float64

        >>> agg = fx.Agg(on="revenue", by=["country", "place"], how=stats.Mean())
        >>> for x in X:
        ...     agg = agg.learn_one(x)
        >>> agg.state
        country  place
        France   Taco Bell      46.0
        Sweden   Burger King    18.0
        France   Burger King    17.0
        Sweden   Taco Bell      69.0
        Name: revenue_mean_by_country_and_place, dtype: float64

        """
        return pd.Series(
            (stat.get() for stat in self._groups.values()),
            index=(
                pd.Index(key[0] for key in self._groups.keys())
                if self.by and len(self.by) == 1
                else pd.MultiIndex.from_tuples(self._groups.keys(), names=self.by)
            ),
            name=self._feature_name,
        )

    def __str__(self):
        return self._feature_name


class TargetAgg(base.SupervisedTransformer, Agg):
    """Computes a streaming aggregate of the target values.

    This transformer is identical to `feature_extraction.Agg`, the only difference is that it
    operates on the target rather than on a feature. At each step, the running statistic `how` of
    target values in group `by` is updated with the target. It is therefore a supervised
    transformer.

    Parameters
    ----------
    by
        The feature by which to group the target values. All the data is included in the aggregate
        if this is `None`.
    how
        The statistic to compute.
    target_name
        The target name which is used in the result.

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
    {'y_bayes_mean_by_place': 3.0}
    {'y_bayes_mean_by_place': 3.0}
    {'y_bayes_mean_by_place': 9.5}
    {'y_bayes_mean_by_place': 22.5}
    {'y_bayes_mean_by_place': 14.333}
    {'y_bayes_mean_by_place': 34.333}
    {'y_bayes_mean_by_place': 15.75}
    {'y_bayes_mean_by_place': 38.25}

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
    {'y_bayes_mean_by_place_and_country': 3.0}
    {'y_bayes_mean_by_place_and_country': 3.0}
    {'y_bayes_mean_by_place_and_country': 3.0}
    {'y_bayes_mean_by_place_and_country': 3.0}
    {'y_bayes_mean_by_place_and_country': 9.5}
    {'y_bayes_mean_by_place_and_country': 22.5}
    {'y_bayes_mean_by_place_and_country': 13.5}
    {'y_bayes_mean_by_place_and_country': 30.5}

    >>> agg.state
    place        country
    Taco Bell    France     31.666667
    Burger King  Sweden     13.000000
                 France     12.333333
    Taco Bell    Sweden     47.000000
    Name: y_bayes_mean_by_place_and_country, dtype: float64

    References
    ----------
    1. [Streaming groupbys in pandas for big datasets](https://maxhalford.github.io/blog/streaming-groupbys-in-pandas-for-big-datasets/)

    """

    def __init__(
        self,
        by: typing.Optional[typing.Union[str, typing.List[str]]],
        how: stats.base.Univariate,
        target_name="y",
    ):
        super().__init__(on=target_name, by=by, how=how)

    @property
    def target_name(self):
        return self.on

    def learn_one(self, x, y):
        self._groups[self._make_key(x)].update(y)
        return self
