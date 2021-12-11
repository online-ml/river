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
