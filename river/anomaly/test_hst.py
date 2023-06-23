from __future__ import annotations


def test_missing_features():
    """Checks that HalfSpaceTrees works even if a feature is missing.

    >>> import random
    >>> from river import anomaly
    >>> from river import compose
    >>> from river import datasets
    >>> from river import metrics
    >>> from river import preprocessing

    >>> model = compose.Pipeline(
    ...     preprocessing.MinMaxScaler(),
    ...     anomaly.HalfSpaceTrees(seed=42)
    ... )

    >>> auc = metrics.ROCAUC()

    >>> features = list(next(iter(datasets.CreditCard()))[0].keys())
    >>> random.seed(42)

    >>> for x, y in datasets.CreditCard().take(8000):
    ...     del x[random.choice(features)]
    ...     score = model.score_one(x)
    ...     model = model.learn_one(x, y)
    ...     auc = auc.update(y, score)

    >>> auc
    ROCAUC: 88.68%

    """
