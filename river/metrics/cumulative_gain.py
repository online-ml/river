from __future__ import annotations

from river import metrics

__all__ = ["CumulativeGain"]


class CumulativeGain(metrics.base.MeanMetric, metrics.base.RankingMetric):
    """Cumulative Gain.

    CG is the sum of the relevances of the first `k` items of a ranking. It ignores the order of
    the items within the first `k` positions. Use `DCG` to reward relevant items that are placed
    near the top.

    The metric is updated once per query, and `get` returns the mean CG over all the queries seen
    so far.

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
    first.

    >>> y_true = [{'a': 3, 'b': 2, 'c': 1}, {'a': 1, 'c': 1}]
    >>> y_pred = [['a', 'b', 'c'], ['b', 'a', 'c']]

    >>> metric = metrics.CumulativeGain()

    >>> for yt, yp in zip(y_true, y_pred):
    ...     metric.update(yt, yp)
    ...     print(metric.get())
    6.0
    4.0

    >>> metric = metrics.CumulativeGain(k=1)
    >>> metric.update({'a': 3, 'b': 2}, ['b', 'a'])
    >>> metric.get()
    2.0

    """

    def __init__(self, k: int | None = None):
        super().__init__()
        if k is not None and k < 1:
            raise ValueError("k must be a positive integer or None")
        self.k = k

    @staticmethod
    def _score(gains) -> float:
        return sum(gains)

    @staticmethod
    def _relevances(y_true) -> dict:
        return y_true if isinstance(y_true, dict) else dict.fromkeys(y_true, 1)

    def _eval(self, y_true, y_pred):
        relevances = self._relevances(y_true)
        ranking = y_pred if self.k is None else y_pred[: self.k]
        return self._score(relevances.get(item, 0) for item in ranking)
