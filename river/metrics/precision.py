from . import base


__all__ = [
    "MacroPrecision",
    "MicroPrecision",
    "Precision",
    "WeightedPrecision",
    "ExamplePrecision",
]


class Precision(base.BinaryMetric):
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
    Precision: 1.
    Precision: 0.5
    Precision: 0.5
    Precision: 0.666667
    Precision: 0.75

    """

    def get(self):
        tp = self.cm.true_positives(self.pos_val)
        fp = self.cm.false_positives(self.pos_val)
        try:
            return tp / (tp + fp)
        except ZeroDivisionError:
            return 0.0


class MacroPrecision(base.MultiClassMetric):
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
    MacroPrecision: 1.
    MacroPrecision: 0.25
    MacroPrecision: 0.5
    MacroPrecision: 0.5
    MacroPrecision: 0.5

    """

    def get(self):
        total = 0
        for c in self.cm.classes:
            try:
                total += self.cm[c][c] / self.cm.sum_col[c]
            except ZeroDivisionError:
                continue
        try:
            return total / len(self.cm.classes)
        except ZeroDivisionError:
            return 0.0


class MicroPrecision(base.MultiClassMetric):
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
    MicroPrecision: 1.
    MicroPrecision: 0.5
    MicroPrecision: 0.666667
    MicroPrecision: 0.75
    MicroPrecision: 0.6

    References
    ----------
    [^1]: [Why are precision, recall and F1 score equal when using micro averaging in a multi-class problem?](https://simonhessner.de/why-are-precision-recall-and-f1-score-equal-when-using-micro-averaging-in-a-multi-class-problem/)

    """

    def get(self):
        num = 0
        den = 0
        for c in self.cm.classes:
            num += self.cm[c][c]
            den += self.cm.sum_col[c]
        try:
            return num / den
        except ZeroDivisionError:
            return 0.0


class WeightedPrecision(base.MultiClassMetric):
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
    WeightedPrecision: 1.
    WeightedPrecision: 0.25
    WeightedPrecision: 0.5
    WeightedPrecision: 0.625
    WeightedPrecision: 0.7

    """

    def get(self):
        total = 0
        for c in self.cm.classes:
            try:
                total += self.cm.sum_row[c] * self.cm[c][c] / self.cm.sum_col[c]
            except ZeroDivisionError:
                continue
        try:
            return total / self.cm.total_weight
        except ZeroDivisionError:
            return 0.0


class ExamplePrecision(base.MultiOutputClassificationMetric):
    """Example-based precision score for multilabel classification.

    Parameters
    ----------
    cm
        This parameter allows sharing the same confusion matrix between multiple metrics. Sharing a
        confusion matrix reduces the amount of storage and computation time.

    Examples
    --------

    >>> from river import metrics

    >>> y_true = [
    ...     {0: False, 1: True, 2: True},
    ...     {0: True, 1: True, 2: False},
    ...     {0: True, 1: True, 2: False},
    ... ]

    >>> y_pred = [
    ...     {0: True, 1: True, 2: True},
    ...     {0: True, 1: False, 2: False},
    ...     {0: True, 1: True, 2: False},
    ... ]

    >>> metric = metrics.ExamplePrecision()
    >>> for yt, yp in zip(y_true, y_pred):
    ...     metric = metric.update(yt, yp)

    >>> metric
    ExamplePrecision: 0.888889

    """

    @property
    def bigger_is_better(self):
        return True

    @property
    def requires_labels(self):
        return True

    def get(self):

        try:
            return self.cm.precision_sum / self.cm.n_samples
        except ZeroDivisionError:
            return 0.0
