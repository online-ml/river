from __future__ import annotations

import math

from river import metrics

__all__ = ["DCG"]


class DCG(metrics.base.MeanMetric, metrics.base.RankingMetric):
    """Discounted Cumulative Gain.

    DCG rewards relevant items that are placed near the top of a ranking. It is the sum of
    `relevance / log2(position + 1)` over the first `k` positions, where positions start at 1.
    The metric is updated once per query, and `get` returns the mean DCG over all the queries seen
    so far.

    DCG is not bounded by 1 and depends on the number of relevant items. Use `NDCG` to compare
    queries with each other.

    Parameters
    ----------
    k
        Only the first `k` items of each ranking are considered. If `None`, the whole ranking is
        used.

    Examples
    --------

    >>> from river import metrics

    Relevances are given as a dict mapping each item to a non-negative grade. Items which are
    absent from the dict are considered irrelevant. Predictions are ranked lists of items, best
    first, and are assumed to contain no duplicates.

    >>> y_true = [{'a': 3, 'b': 2, 'c': 1}, {'a': 1, 'c': 1}]
    >>> y_pred = [['a', 'b', 'c'], ['b', 'a', 'c']]

    >>> metric = metrics.DCG()

    >>> for yt, yp in zip(y_true, y_pred):
    ...     metric.update(yt, yp)
    ...     print(metric.get())
    4.7619
    2.9464

    """

    def __init__(self, k: int | None = None):
        super().__init__()
        if k is not None and k < 1:
            raise ValueError("k must be a positive integer or None")
        self.k = k

    @staticmethod
    def _dcg(gains) -> float:
        return sum(g / math.log2(i + 2) for i, g in enumerate(gains))

    @staticmethod
    def _relevances(y_true) -> dict:
        return y_true if isinstance(y_true, dict) else dict.fromkeys(y_true, 1)

    def _eval(self, y_true, y_pred):
        relevances = self._relevances(y_true)
        ranking = y_pred if self.k is None else y_pred[: self.k]
        return self._dcg(relevances.get(item, 0) for item in ranking)
