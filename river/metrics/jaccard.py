from __future__ import annotations

from river import metrics

__all__ = ["Jaccard", "MacroJaccard", "MicroJaccard", "WeightedJaccard"]


class Jaccard(metrics.base.BinaryMetric):
    """Jaccard score.

    Parameters
    ----------
    cm
        This parameter allows sharing the same confusion matrix between multiple metrics. Sharing
        a confusion matrix reduces the amount of storage and computation time.
    pos_val
        Value to treat as "positive".

    Examples
    --------

    >>> from river import metrics

    >>> y_true = [False, True, True]
    >>> y_pred = [True, True, True]

    >>> metric = metrics.Jaccard()

    >>> for yt, yp in zip(y_true, y_pred):
    ...     print(metric.update(yt, yp))
    Jaccard: 0.00%
    Jaccard: 50.00%
    Jaccard: 66.67%

    References
    ----------
    [^1]: [Jaccard index](https://www.wikiwand.com/en/Jaccard_index)

    """

    def get(self):
        tp = self.cm.true_positives(self.pos_val)
        fp = self.cm.false_positives(self.pos_val)
        fn = self.cm.false_negatives(self.pos_val)
        try:
            return tp / (tp + fp + fn)
        except ZeroDivisionError:
            return 0.0


class MacroJaccard(metrics.base.MultiClassMetric):
    """Macro-average Jaccard score.

    Parameters
    ----------
    cm
        This parameter allows sharing the same confusion
        matrix between multiple metrics. Sharing a confusion matrix reduces the amount of storage
        and computation time.

    Examples
    --------

    >>> from river import metrics

    >>> y_true = [0, 1, 2, 2, 2]
    >>> y_pred = [0, 0, 2, 2, 1]

    >>> metric = metrics.MacroJaccard()

    >>> for yt, yp in zip(y_true, y_pred):
    ...     print(metric.update(yt, yp))
    MacroJaccard: 100.00%
    MacroJaccard: 25.00%
    MacroJaccard: 50.00%
    MacroJaccard: 50.00%
    MacroJaccard: 38.89%

    """

    def get(self):
        total = 0
        for c in self.cm.classes:
            try:
                tp = self.cm.true_positives(c)
                fp = self.cm.false_positives(c)
                fn = self.cm.false_negatives(c)
                total += tp / (tp + fp + fn)
            except ZeroDivisionError:
                continue
        try:
            return total / len(self.cm.classes)
        except ZeroDivisionError:
            return 0.0


class MicroJaccard(metrics.base.MultiClassMetric):
    """Micro-average Jaccard score.

    Parameters
    ----------
    cm
        This parameter allows sharing the same confusion
        matrix between multiple metrics. Sharing a confusion matrix reduces the amount of storage
        and computation time.

    Examples
    --------

    >>> from river import metrics

    >>> y_true = [0, 1, 2, 2, 2]
    >>> y_pred = [0, 0, 2, 2, 1]

    >>> metric = metrics.MicroJaccard()

    >>> for yt, yp in zip(y_true, y_pred):
    ...     print(metric.update(yt, yp))
    MicroJaccard: 100.00%
    MicroJaccard: 33.33%
    MicroJaccard: 50.00%
    MicroJaccard: 60.00%
    MicroJaccard: 42.86%

    """

    def get(self):
        tp = self.cm.total_true_positives
        fp = self.cm.total_false_positives
        fn = self.cm.total_false_negatives

        try:
            return tp / (tp + fp + fn)
        except ZeroDivisionError:
            return 0.0


class WeightedJaccard(metrics.base.MultiClassMetric):
    """Weighted average Jaccard score.

    Parameters
    ----------
    cm
        This parameter allows sharing the same confusion
        matrix between multiple metrics. Sharing a confusion matrix reduces the amount of storage
        and computation time.

    Examples
    --------

    >>> from river import metrics

    >>> y_true = [0, 1, 2, 2, 2]
    >>> y_pred = [0, 0, 2, 2, 1]

    >>> metric = metrics.WeightedJaccard()

    >>> for yt, yp in zip(y_true, y_pred):
    ...     print(metric.update(yt, yp))
    WeightedJaccard: 100.00%
    WeightedJaccard: 25.00%
    WeightedJaccard: 50.00%
    WeightedJaccard: 62.50%
    WeightedJaccard: 50.00%

    """

    def get(self):
        total = 0
        for c in self.cm.classes:
            try:
                tp = self.cm.true_positives(c)
                fp = self.cm.false_positives(c)
                fn = self.cm.false_negatives(c)
                total += self.cm.support(c) * tp / (tp + fp + fn)
            except ZeroDivisionError:
                continue
        try:
            return total / self.cm.total_weight
        except ZeroDivisionError:
            return 0.0
