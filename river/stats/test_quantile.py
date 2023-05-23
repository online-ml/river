from __future__ import annotations

import random

from river import stats


def test_issue_1178():
    """

    https://github.com/online-ml/river/issues/1178

    >>> from river import stats

    >>> q = stats.Quantile(0.01)
    >>> for x in [5, 0, 0, 0, 0, 0, 0, 0]:
    ...     q = q.update(x)
    ...     print(q)
    Quantile: 5.
    Quantile: 0.
    Quantile: 0.
    Quantile: 0.
    Quantile: 0.
    Quantile: 0.
    Quantile: 0.
    Quantile: 0.

    >>> q = stats.Quantile(0.99)
    >>> for x in [5, 0, 0, 0, 0, 0, 0, 0]:
    ...     q = q.update(x)
    ...     print(q)
    Quantile: 5.
    Quantile: 5.
    Quantile: 5.
    Quantile: 5.
    Quantile: 5.
    Quantile: 0.
    Quantile: 0.277778
    Quantile: 0.827546

    """


def test_ge():
    low = stats.Quantile(0.01)
    high = stats.Quantile(0.99)

    for _ in range(100):
        x = random.random()
        low.update(x)
        high.update(x)
        assert high.get() >= low.get()
