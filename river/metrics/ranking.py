from __future__ import annotations

import abc
import collections
import typing

from river import metrics

__all__ = [
    "PrecisionAtK",
    "RecallAtK",
    "F1AtK",
]


class BinaryRankAtKMetric(metrics.base.MeanMetric, metrics.base.RankingMetric):
    """Helper functions for the binary ranking metrics

    Parameters
    ----------
    relevance_threshold
        relevance threshold to calculate relevant items in case relevance scores are provided for a binary ranking
        problem. If y_true score >= relevance_threshold then the item is relevant.
    """

    def __init__(self, k=None, relevance_threshold=None):
        metrics.base.MeanMetric.__init__(self)
        metrics.base.RankingMetric.__init__(self, k=k)
        self.relevance_threshold = relevance_threshold

    def _relevance(
        self, y_true: dict[typing.Hashable, float] | list[typing.Hashable]
    ) -> list[typing.Hashable]:
        if isinstance(y_true, collections.abc.Mapping) and self.relevance_threshold is None:
            raise ValueError("relevance_threshold is needed when passing a dictionary of scores")
        if isinstance(y_true, collections.abc.Mapping):
            return [i for i, v in y_true.items() if v >= self.relevance_threshold]
        else:
            return y_true


class PrecisionAtK(BinaryRankAtKMetric):
    """Precision@k for ranking

    The metric quantifies how many items in the top-K results are relevant. It is calculated as
    TP@k / (TP@k + FP@k)

    Parameters
    ----------
    k
        only consider the highest k scores in the ranking.
    relevance_threshold
        relevance threshold to calculate relevant items in case relevance scores are provided for a binary ranking
        problem. If y_true score >= relevance_threshold then the item is relevant.

    Examples
    --------
    >>> from river import metrics

    >>> metric = metrics.PrecisionAtK(k=2)

    >>> y_true = [['Harry Potter','Once upon a time in Hollywood'],
    ...           ['Harry Potter','Once upon a time in Hollywood']]
    >>> y_pred = [['Once upon a time in Hollywood'],['Harry Potter','Once upon a time in Hollywood']]

    >>> for yt, yp in zip(y_true, y_pred):
    ...    metric.update(yt, yp)
    ...    print(metric)
    PrecisionAtK: 50.00%
    PrecisionAtK: 75.00%

    Notes
    -----
    - if len(y_pred) < k, k is still used to calculate metric
    - if len(y_true) < k, P@k is bounded above by len(y_true)/k
    - y_true may be a list/set of relevant items, or a dict of scores used with relevance_threshold
    - y_pred is a ranked list of predicted items, ordered by descending predicted relevance
    """

    def _eval(self, y_true, y_pred):
        y_true = self._relevance(y_true)
        tmp_k = self._resolve_k(y_pred)
        y_pred = y_pred[:tmp_k]
        try:
            return len(set(y_pred) & set(y_true)) / tmp_k
        except ZeroDivisionError:
            return 0


class RecallAtK(BinaryRankAtKMetric):
    """Recall@k for ranking

    The metric quantifies how many relevant results were shown out of all relevant results for the query.
    It is calculated as TP@k / (TP@k + FN@k)

    Parameters
    ----------
    k
        only consider the highest k scores in the ranking.
    relevance_threshold
        relevance threshold to calculate relevant items in case relevance scores are provided for a binary ranking
        problem. If y_true score >= relevance_threshold then the item is relevant.

    Examples
    --------
    >>> from river import metrics

    >>> metric = metrics.RecallAtK(k=2)

    >>> y_true = [['Harry Potter','Once upon a time in Hollywood'],
    ...           ['Harry Potter','Once upon a time in Hollywood']]
    >>> y_pred = [['Once upon a time in Hollywood'],['Harry Potter','Once upon a time in Hollywood']]

    >>> for yt, yp in zip(y_true, y_pred):
    ...    metric.update(yt, yp)
    ...    print(metric)
    RecallAtK: 50.00%
    RecallAtK: 75.00%
    
    Notes
    -----
    - y_true may be a list/set of relevant items, or a dict of scores used with relevance_threshold
    - y_pred is a ranked list of predicted items, ordered by descending predicted relevance
    """

    def _eval(self, y_true, y_pred):
        y_true = self._relevance(y_true)
        tmp_k = self._resolve_k(y_pred)
        y_pred = y_pred[:tmp_k]
        try:
            return len(set(y_pred) & set(y_true)) / len(set(y_true))
        except ZeroDivisionError:
            return 0


class F1AtK(BinaryRankAtKMetric):
    """F1@k for ranking

    F1 metric that combines PrecisionAtK and RecallAtK.
    It is calculated as a version of the 2TP/(2TP + FP + FN) formula so that zero division is
    inherently addressed: 2TP / (k + n_of_all_relevant)

    Parameters
    ----------
    k
        only consider the highest k scores in the ranking.
    relevance_threshold
        relevance threshold to calculate relevant items in case relevance scores are provided for a binary ranking
        problem. If y_true score >= relevance_threshold then the item is relevant.

    Examples
    --------
    >>> from river import metrics

    >>> metric = metrics.F1AtK(k=2)

    >>> y_true = [['Harry Potter','Once upon a time in Hollywood'],
    ...           ['Harry Potter','Once upon a time in Hollywood']]
    >>> y_pred = [['Once upon a time in Hollywood'],['Harry Potter','Once upon a time in Hollywood']]

    >>> for yt, yp in zip(y_true, y_pred):
    ...    metric.update(yt, yp)
    ...    print(metric)
    F1AtK: 50.00%
    F1AtK: 75.00%
    
    Notes
    -----
    - y_true may be a list/set of relevant items, or a dict of scores used with relevance_threshold
    - y_pred is a ranked list of predicted items, ordered by descending predicted relevance
    """

    def _eval(self, y_true, y_pred):
        y_true = self._relevance(y_true)
        tmp_k = self._resolve_k(y_pred)
        y_pred = y_pred[:tmp_k]
        try:
            return (2 * len(set(y_pred) & set(y_true))) / (tmp_k + len(set(y_true)))
        except ZeroDivisionError:
            return 0