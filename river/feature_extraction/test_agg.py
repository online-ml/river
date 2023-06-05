from __future__ import annotations


def test_agg_lag():
    """

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

    >>> from river.feature_extraction import Agg
    >>> from river.stats import Shift

    >>> agg = (
    ...     Agg("customers", None, Shift(2)) +
    ...     Agg("revenue", None, Shift(2))
    ... )
    >>> for x in X:
    ...     print(agg.learn_one(x).transform_one(x))
    {'revenue_shift_2': None, 'customers_shift_2': None}
    {'revenue_shift_2': None, 'customers_shift_2': None}
    {'revenue_shift_2': 420, 'customers_shift_2': 10}
    {'revenue_shift_2': 386, 'customers_shift_2': 10}
    {'revenue_shift_2': 145, 'customers_shift_2': 20}
    {'revenue_shift_2': 87, 'customers_shift_2': 5}
    {'revenue_shift_2': 276, 'customers_shift_2': 15}
    {'revenue_shift_2': 149, 'customers_shift_2': 10}
    {'revenue_shift_2': 890, 'customers_shift_2': 30}
    {'revenue_shift_2': 782, 'customers_shift_2': 40}

    We can also specify multiple lags for a given feature.

    >>> from river.compose import TransformerUnion

    >>> agg = TransformerUnion(*[
    ...     Agg("customers", None, Shift(d))
    ...     for d in [1, 2, 3]
    ... ])
    >>> for x in X:
    ...     print(agg.learn_one(x).transform_one(x))
    {'customers_shift_3': None, 'customers_shift_2': None, 'customers_shift_1': None}
    {'customers_shift_3': None, 'customers_shift_2': None, 'customers_shift_1': 10}
    {'customers_shift_3': None, 'customers_shift_2': 10, 'customers_shift_1': 10}
    {'customers_shift_3': 10, 'customers_shift_2': 10, 'customers_shift_1': 20}
    {'customers_shift_3': 10, 'customers_shift_2': 20, 'customers_shift_1': 5}
    {'customers_shift_3': 20, 'customers_shift_2': 5, 'customers_shift_1': 15}
    {'customers_shift_3': 5, 'customers_shift_2': 15, 'customers_shift_1': 10}
    {'customers_shift_3': 15, 'customers_shift_2': 10, 'customers_shift_1': 30}
    {'customers_shift_3': 10, 'customers_shift_2': 30, 'customers_shift_1': 40}
    {'customers_shift_3': 30, 'customers_shift_2': 40, 'customers_shift_1': 20}

    As of now, we're looking at lagged values over all the data. It might make more sense to look
    at the lags per shop.

    >>> agg = Agg("customers", "shop", Shift(1))
    >>> for x in X:
    ...     print(agg.learn_one(x).transform_one(x))
    {'customers_shift_1_by_shop': None}
    {'customers_shift_1_by_shop': None}
    {'customers_shift_1_by_shop': 10}
    {'customers_shift_1_by_shop': 10}
    {'customers_shift_1_by_shop': 20}
    {'customers_shift_1_by_shop': 5}
    {'customers_shift_1_by_shop': 15}
    {'customers_shift_1_by_shop': 10}
    {'customers_shift_1_by_shop': 30}
    {'customers_shift_1_by_shop': 40}

    """


def test_target_agg_lag():
    """

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

    >>> from river.feature_extraction import TargetAgg
    >>> from river.stats import Shift

    >>> agg = TargetAgg(None, Shift(1)) + TargetAgg(None, Shift(2))
    >>> for x, y in dataset:
    ...     print(agg.transform_one(x))
    ...     agg = agg.learn_one(x, y)
    {'y_shift_2': None, 'y_shift_1': None}
    {'y_shift_2': None, 'y_shift_1': None}
    {'y_shift_2': None, 'y_shift_1': 42}
    {'y_shift_2': 42, 'y_shift_1': 16}
    {'y_shift_2': 16, 'y_shift_1': 24}
    {'y_shift_2': 24, 'y_shift_1': 58}
    {'y_shift_2': 58, 'y_shift_1': 20}
    {'y_shift_2': 20, 'y_shift_1': 50}

    We can also calculate the lags with different groups:

    >>> agg = TargetAgg("country", Shift(1)) + TargetAgg("country", Shift(2))
    >>> for x, y in dataset:
    ...     print(agg.transform_one(x))
    ...     agg = agg.learn_one(x, y)
    {'y_shift_2_by_country': None, 'y_shift_1_by_country': None}
    {'y_shift_2_by_country': None, 'y_shift_1_by_country': None}
    {'y_shift_2_by_country': None, 'y_shift_1_by_country': None}
    {'y_shift_2_by_country': None, 'y_shift_1_by_country': None}
    {'y_shift_2_by_country': None, 'y_shift_1_by_country': 16}
    {'y_shift_2_by_country': None, 'y_shift_1_by_country': 42}
    {'y_shift_2_by_country': 42, 'y_shift_1_by_country': 24}
    {'y_shift_2_by_country': 16, 'y_shift_1_by_country': 58}

    """
