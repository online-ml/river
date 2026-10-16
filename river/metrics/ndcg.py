from __future__ import annotations

from river import metrics

__all__ = ["NDCG"]


class NDCG(metrics.dcg.DCG):
    """Normalized Discounted Cumulative Gain.

    NDCG is DCG divided by the DCG of the ideal ranking (IDCG), so that the score lies between 0
    and 1. A query for which no item is relevant scores 0.

    Parameters
    ----------
    k
        Only the first `k` items of each ranking are considered. If `None`, the whole ranking is
        used.

    Examples
    --------

    >>> from river import metrics

    >>> y_true = [
    ...     {'a': 3, 'b': 2, 'c': 3, 'd': 0, 'e': 1, 'f': 2},
    ...     {'a': 1, 'b': 0, 'c': 0},
    ...     {'a': 1, 'c': 1},
    ... ]
    >>> y_pred = [
    ...     ['a', 'b', 'c', 'd', 'e', 'f'],
    ...     ['b', 'c', 'a'],
    ...     ['b', 'a', 'c'],
    ... ]

    >>> metric = metrics.NDCG()

    >>> for yt, yp in zip(y_true, y_pred):
    ...     metric.update(yt, yp)
    ...     print(metric.get())
    0.9608
    0.7304
    0.7181

    >>> metric
    NDCG: 0.7181

    A list or set of items can be given instead of a dict, in which case every listed item has a
    relevance of 1. The ranking can also be truncated with `k`.

    >>> metric = metrics.NDCG(k=2)
    >>> metric.update({'a', 'c'}, ['b', 'a', 'c'])
    >>> metric.get()
    0.3869

    """

    def _eval(self, y_true, y_pred):
        dcg = super()._eval(y_true, y_pred)
        ideal = sorted(self._relevances(y_true).values(), reverse=True)[: self.k]
        idcg = self._dcg(ideal)
        return dcg / idcg if idcg > 0 else 0.0
