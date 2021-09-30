from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

from river import base, utils


class Lagger(base.Transformer):
    """Uses lagged values as features.

    Parameters
    ----------
    by
        An optional feature by which to group the lagged values.
    drop_nones
        Whether or not features should be included with a `None` value if not enough values have
        been seen yet.

    Examples
    --------

    Let's say we have daily data about the number of customers and the revenue for two shops.

    >>> X = [
    ...     {'shop': '7/11', 'customers': 10, 'revenue': 420},
    ...     {'shop': 'Kmart', 'customers': 10, 'revenue': 386},
    ...     {'shop': '7/11', 'customers': 20, 'revenue': 145},
    ...     {'shop': 'Kmart', 'customers': 5, 'revenue': 87},
    ...     {'shop': '7/11', 'customers': 15, 'revenue': 276},
    ...     {'shop': 'Kmart', 'customers': 10, 'revenue': 149},
    ...     {'shop': '7/11', 'customers': 30, 'revenue': 890},
    ...     {'shop': 'Kmart', 'customers': 40, 'revenue': 782},
    ...     {'shop': '7/11', 'customers': 20, 'revenue': 403},
    ...     {'shop': 'Kmart', 'customers': 35, 'revenue': 218},
    ... ]

    At each time step, we want to use the number of customers and revenue from 2 time steps ago:

    >>> from river.feature_extraction import Lagger

    >>> lagger = Lagger(customers=2, revenue=2)
    >>> for x in X:
    ...     lagger = lagger.learn_one(x)
    ...     print(lagger.transform_one(x))
    {}
    {}
    {'customers-2': 10, 'revenue-2': 420}
    {'customers-2': 10, 'revenue-2': 386}
    {'customers-2': 20, 'revenue-2': 145}
    {'customers-2': 5, 'revenue-2': 87}
    {'customers-2': 15, 'revenue-2': 276}
    {'customers-2': 10, 'revenue-2': 149}
    {'customers-2': 30, 'revenue-2': 890}
    {'customers-2': 40, 'revenue-2': 782}

    We can also specify multiple lags for a given feature.

    >>> lagger = Lagger(customers=[1, 2, 3])
    >>> for x in X:
    ...     lagger = lagger.learn_one(x)
    ...     print(lagger.transform_one(x))
    {}
    {'customers-1': 10}
    {'customers-1': 10, 'customers-2': 10}
    {'customers-1': 20, 'customers-2': 10, 'customers-3': 10}
    {'customers-1': 5, 'customers-2': 20, 'customers-3': 10}
    {'customers-1': 15, 'customers-2': 5, 'customers-3': 20}
    {'customers-1': 10, 'customers-2': 15, 'customers-3': 5}
    {'customers-1': 30, 'customers-2': 10, 'customers-3': 15}
    {'customers-1': 40, 'customers-2': 30, 'customers-3': 10}
    {'customers-1': 20, 'customers-2': 40, 'customers-3': 30}

    As of now, we're looking at lagged values over all the data. It might make more sense to look
    at the lags per shop.

    >>> lagger = Lagger(customers=1, by='shop')
    >>> for x in X:
    ...     lagger = lagger.learn_one(x)
    ...     print(lagger.transform_one(x))
    {}
    {}
    {'customers-1_by_shop': 10}
    {'customers-1_by_shop': 10}
    {'customers-1_by_shop': 20}
    {'customers-1_by_shop': 5}
    {'customers-1_by_shop': 15}
    {'customers-1_by_shop': 10}
    {'customers-1_by_shop': 30}
    {'customers-1_by_shop': 40}

    Past values are stored in a window. At the beginning, these windows are empty. By default,
    features are omitted if the window is not long enough. You can also include them with a value
    of `None`:

    >>> lagger = Lagger(drop_nones=False, customers=1, by='shop')
    >>> for x in X:
    ...     lagger = lagger.learn_one(x)
    ...     print(lagger.transform_one(x))
    {'customers-1_by_shop': None}
    {'customers-1_by_shop': None}
    {'customers-1_by_shop': 10}
    {'customers-1_by_shop': 10}
    {'customers-1_by_shop': 20}
    {'customers-1_by_shop': 5}
    {'customers-1_by_shop': 15}
    {'customers-1_by_shop': 10}
    {'customers-1_by_shop': 30}
    {'customers-1_by_shop': 40}

    Note that lags can also be specified with a `dict`:

    >>> lags = {'customers': 2, 'revenue': 2}
    >>> lagger = Lagger(**lags)

    """

    def __init__(
        self,
        by: Optional[Union[str, List[str]]] = None,
        drop_nones=True,
        **lags: Dict[str, Union[int, Tuple[int]]],
    ):

        for feature, size in lags.items():
            if isinstance(size, int):
                lags[feature] = [size]

        if by is not None and not isinstance(by, list):
            by = [by]

        self.by = by
        self.drop_nones = drop_nones
        self.lags = lags

        self._windows = defaultdict(
            lambda: {
                feature: utils.Window(size=max(sizes) + 1)
                for feature, sizes in self.lags.items()
            }
        )

    def _make_key(self, x):
        if self.by:
            return tuple(x[k] for k in self.by)
        return None

    def learn_one(self, x):

        key = self._make_key(x)

        for feature, window in self._windows[key].items():
            window.append(x[feature])

        return self

    def _make_feature_name(self, feature: str, size: int) -> str:
        name = f"{feature}-{size}"
        if self.by:
            name += "_by_" + "_and_".join(self.by)
        return name

    def transform_one(self, x):
        xt = {}

        key = self._make_key(x)
        windows = self._windows[key]

        for feature, sizes in self.lags.items():
            window = windows[feature]
            for size in sizes:
                try:
                    lag = window[-size - 1]
                except IndexError:
                    if self.drop_nones:
                        continue
                    lag = None
                xt[self._make_feature_name(feature, size)] = lag

        return xt


class TargetLagger(base.SupervisedTransformer):
    """Uses lagged values of the target as features.

    Parameters
    ----------
    by
        An optional feature by which to group the lagged values.
    drop_nones
        Whether or not features should be included with a `None` value if not enough values have
        been seen yet.
    target_name
        The target name which is used in the result.

    Examples
    --------

    Consider the following dataset, where the second value of each value is the target:

    >>> dataset = [
    ...     ({'country': 'France'}, 42),
    ...     ({'country': 'Sweden'}, 16),
    ...     ({'country': 'France'}, 24),
    ...     ({'country': 'Sweden'}, 58),
    ...     ({'country': 'Sweden'}, 20),
    ...     ({'country': 'France'}, 50),
    ...     ({'country': 'France'}, 10),
    ...     ({'country': 'Sweden'}, 80)
    ... ]

    Let's extract the two last values of the target at each time step.

    >>> from river.feature_extraction import TargetLagger

    >>> lagger = TargetLagger(lags=[1, 2])
    >>> for x, y in dataset:
    ...     print(lagger.transform_one(x))
    ...     lagger = lagger.learn_one(x, y)
    {}
    {}
    {'target-1': 42}
    {'target-1': 16, 'target-2': 42}
    {'target-1': 24, 'target-2': 16}
    {'target-1': 58, 'target-2': 24}
    {'target-1': 20, 'target-2': 58}
    {'target-1': 50, 'target-2': 20}

    We can also calculate the lags with different groups:

    >>> lagger = TargetLagger(lags=[1, 2], by=['country'])
    >>> for x, y in dataset:
    ...     print(lagger.transform_one(x))
    ...     lagger = lagger.learn_one(x, y)
    {}
    {}
    {}
    {}
    {'target-1_by_country': 16}
    {'target-1_by_country': 42}
    {'target-1_by_country': 24, 'target-2_by_country': 42}
    {'target-1_by_country': 58, 'target-2_by_country': 16}

    """

    def __init__(
        self,
        lags: Union[int, Tuple[int]],
        by: Optional[Union[str, List[str]]] = None,
        drop_nones=True,
        target_name="target",
    ):

        if isinstance(lags, int):
            lags = [lags]

        if by is not None and not isinstance(by, list):
            by = [by]

        self.by = by
        self.drop_nones = drop_nones
        self.lags = lags
        self.target_name = target_name

        self._windows = defaultdict(lambda: utils.Window(size=max(lags) + 1))

    def _make_key(self, x):
        if self.by:
            return tuple(x[k] for k in self.by)
        return None

    def learn_one(self, x, y):
        self._windows[self._make_key(x)].append(y)
        return self

    def _make_feature_name(self, size: int) -> str:
        name = f"{self.target_name}-{size}"
        if self.by:
            name += "_by_" + "_and_".join(self.by)
        return name

    def transform_one(self, x):
        xt = {}

        key = self._make_key(x)
        window = self._windows[key]

        for size in self.lags:
            try:
                lag = window[-size - 1]
            except IndexError:
                if self.drop_nones:
                    continue
                lag = None
            xt[self._make_feature_name(size)] = lag

        return xt
