from river import metrics

__all__ = ["MacroPrecision", "MicroPrecision", "Precision", "WeightedPrecision"]


class Precision(metrics.base.BinaryMetric):
    """Binary precision score.

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

    >>> y_true = [True, False, True, True, True]
    >>> y_pred = [True, True, False, True, True]

    >>> metric = metrics.Precision()

    >>> for yt, yp in zip(y_true, y_pred):
    ...     print(metric.update(yt, yp))
    Precision: 100.00%
    Precision: 50.00%
    Precision: 50.00%
    Precision: 66.67%
    Precision: 75.00%

    """

    def get(self):
        tp = self.cm.true_positives(self.pos_val)
        fp = self.cm.false_positives(self.pos_val)
        try:
            return tp / (tp + fp)
        except ZeroDivisionError:
            return 0.0


class MacroPrecision(metrics.base.MultiClassMetric):
    """Macro-average precision score.

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

    >>> metric = metrics.MacroPrecision()

    >>> for yt, yp in zip(y_true, y_pred):
    ...     print(metric.update(yt, yp))
    MacroPrecision: 100.00%
    MacroPrecision: 25.00%
    MacroPrecision: 50.00%
    MacroPrecision: 50.00%
    MacroPrecision: 50.00%

    """

    def get(self):
        total = 0
        for c in self.cm.classes:
            try:
                tp = self.cm.true_positives(c)
                fp = self.cm.false_positives(c)
                total += tp / (tp + fp)
            except ZeroDivisionError:
                continue
        try:
            return total / len(self.cm.classes)
        except ZeroDivisionError:
            return 0.0


class MicroPrecision(metrics.base.MultiClassMetric):
    """Micro-average precision score.

    The micro-average precision score is exactly equivalent to the micro-average recall as well as
    the micro-average F1 score.

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

    >>> metric = metrics.MicroPrecision()

    >>> for yt, yp in zip(y_true, y_pred):
    ...     print(metric.update(yt, yp))
    MicroPrecision: 100.00%
    MicroPrecision: 50.00%
    MicroPrecision: 66.67%
    MicroPrecision: 75.00%
    MicroPrecision: 60.00%

    References
    ----------
    [^1]: [Why are precision, recall and F1 score equal when using micro averaging in a multi-class problem?](https://simonhessner.de/why-are-precision-recall-and-f1-score-equal-when-using-micro-averaging-in-a-multi-class-problem/)

    """

    def get(self):
        tp = self.cm.total_true_positives
        fp = self.cm.total_false_positives
        try:
            return tp / (tp + fp)
        except ZeroDivisionError:
            return 0.0


class WeightedPrecision(metrics.base.MultiClassMetric):
    """Weighted-average precision score.

    This uses the support of each label to compute an average score, whereas
    `metrics.MacroPrecision` ignores the support.

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

    >>> metric = metrics.WeightedPrecision()

    >>> for yt, yp in zip(y_true, y_pred):
    ...     print(metric.update(yt, yp))
    WeightedPrecision: 100.00%
    WeightedPrecision: 25.00%
    WeightedPrecision: 50.00%
    WeightedPrecision: 62.50%
    WeightedPrecision: 70.00%

    """

    def get(self):
        total = 0
        for c in self.cm.classes:
            try:
                tp = self.cm.true_positives(c)
                fp = self.cm.false_positives(c)
                total += self.cm.support(c) * tp / (tp + fp)
            except ZeroDivisionError:
                continue
        try:
            return total / self.cm.total_weight
        except ZeroDivisionError:
            return 0.0
