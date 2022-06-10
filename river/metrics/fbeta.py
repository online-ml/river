import collections
import functools

from river import metrics

__all__ = [
    "F1",
    "FBeta",
    "MacroF1",
    "MacroFBeta",
    "MicroF1",
    "MicroFBeta",
    "MultiFBeta",
    "WeightedF1",
    "WeightedFBeta",
]


class FBeta(metrics.base.BinaryMetric):
    """Binary F-Beta score.

    The FBeta score is a weighted harmonic mean between precision and recall. The higher the
    `beta` value, the higher the recall will be taken into account. When `beta` equals 1,
    precision and recall and equivalently weighted, which results in the F1 score (see
    `metrics.F1`).

    Parameters
    ----------
    beta
        Weight of precision in the harmonic mean.
    cm
        This parameter allows sharing the same confusion
        matrix between multiple metrics. Sharing a confusion matrix reduces the amount of storage
        and computation time.
    pos_val
        Value to treat as "positive".

    Attributes
    ----------
    precision : metrics.Precision
    recall : metrics.Recall

    Examples
    --------

    >>> from river import metrics

    >>> y_true = [False, False, False, True, True, True]
    >>> y_pred = [False, False, True, True, False, False]

    >>> metric = metrics.FBeta(beta=2)
    >>> for yt, yp in zip(y_true, y_pred):
    ...     metric = metric.update(yt, yp)

    >>> metric
    FBeta: 35.71%

    """

    def __init__(self, beta: float, cm=None, pos_val=True):
        super().__init__(cm, pos_val)
        self.beta = beta
        self.precision = metrics.Precision(self.cm, self.pos_val)
        self.recall = metrics.Recall(self.cm, self.pos_val)

    def get(self):
        p = self.precision.get()
        r = self.recall.get()
        b2 = self.beta**2
        try:
            return (1 + b2) * p * r / (b2 * p + r)
        except ZeroDivisionError:
            return 0.0


class MacroFBeta(metrics.base.MultiClassMetric):
    """Macro-average F-Beta score.

    This works by computing the F-Beta score per class, and then performs a global average.

    Parameters
    ----------
    beta
        Weight of precision in harmonic mean.
    cm
        This parameter allows sharing the same confusion
        matrix between multiple metrics. Sharing a confusion matrix reduces the amount of storage
        and computation time.

    Examples
    --------

    >>> from river import metrics

    >>> y_true = [0, 1, 2, 2, 2]
    >>> y_pred = [0, 0, 2, 2, 1]

    >>> metric = metrics.MacroFBeta(beta=.8)

    >>> for yt, yp in zip(y_true, y_pred):
    ...     print(metric.update(yt, yp))
    MacroFBeta: 100.00%
    MacroFBeta: 31.06%
    MacroFBeta: 54.04%
    MacroFBeta: 54.04%
    MacroFBeta: 48.60%

    """

    def __init__(self, beta, cm=None):
        super().__init__(cm)
        self.beta = beta

    def get(self):
        total = 0
        b2 = self.beta**2

        for c in self.cm.classes:

            try:
                p = self.cm[c][c] / self.cm.sum_col[c]
            except ZeroDivisionError:
                p = 0

            try:
                r = self.cm[c][c] / self.cm.sum_row[c]
            except ZeroDivisionError:
                r = 0

            try:
                total += (1 + b2) * p * r / (b2 * p + r)
            except ZeroDivisionError:
                continue

        try:
            return total / len(self.cm.classes)
        except ZeroDivisionError:
            return 0.0


class MicroFBeta(metrics.base.MultiClassMetric):
    """Micro-average F-Beta score.

    This computes the F-Beta score by merging all the predictions and true labels, and then
    computes a global F-Beta score.

    Parameters
    ----------
    beta
        Weight of precision in the harmonic mean.
    cm
        This parameter allows sharing the same confusion
        matrix between multiple metrics. Sharing a confusion matrix reduces the amount of storage
        and computation time.

    Examples
    --------

    >>> from river import metrics

    >>> y_true = [0, 1, 2, 2, 0]
    >>> y_pred = [0, 1, 1, 2, 1]

    >>> metric = metrics.MicroFBeta(beta=2)
    >>> for yt, yp in zip(y_true, y_pred):
    ...     metric = metric.update(yt, yp)

    >>> metric
    MicroFBeta: 60.00%

    References
    ----------
    1. [Why are precision, recall and F1 score equal when using micro averaging in a multi-class problem?](https://simonhessner.de/why-are-precision-recall-and-f1-score-equal-when-using-micro-averaging-in-a-multi-class-problem/)

    """

    def __init__(self, beta: float, cm=None):
        super().__init__(cm)
        self.beta = beta
        self.precision = metrics.MicroPrecision(self.cm)
        self.recall = metrics.MicroRecall(self.cm)

    def get(self):
        p = self.precision.get()
        r = self.recall.get()
        b2 = self.beta**2
        try:
            return (1 + b2) * p * r / (b2 * p + r)
        except ZeroDivisionError:
            return 0.0


class WeightedFBeta(metrics.base.MultiClassMetric):
    """Weighted-average F-Beta score.

    This works by computing the F-Beta score per class, and then performs a global weighted average
    according to the support of each class.

    Parameters
    ----------
    beta
        Weight of precision in the harmonic mean.
    cm
        This parameter allows sharing the same confusion
        matrix between multiple metrics. Sharing a confusion matrix reduces the amount of storage
        and computation time.

    Examples
    --------

    >>> from river import metrics

    >>> y_true = [0, 1, 2, 2, 2]
    >>> y_pred = [0, 0, 2, 2, 1]

    >>> metric = metrics.WeightedFBeta(beta=0.8)

    >>> for yt, yp in zip(y_true, y_pred):
    ...     print(metric.update(yt, yp))
    WeightedFBeta: 100.00%
    WeightedFBeta: 31.06%
    WeightedFBeta: 54.04%
    WeightedFBeta: 65.53%
    WeightedFBeta: 62.63%

    """

    def __init__(self, beta, cm=None):
        super().__init__(cm)
        self.beta = beta

    def get(self):
        total = 0
        b2 = self.beta**2

        for c in self.cm.classes:

            try:
                p = self.cm.sum_row[c] * self.cm[c][c] / self.cm.sum_col[c]
            except ZeroDivisionError:
                p = 0

            try:
                r = self.cm.sum_row[c] * self.cm[c][c] / self.cm.sum_row[c]
            except ZeroDivisionError:
                r = 0

            try:
                total += (1 + b2) * p * r / (b2 * p + r)
            except ZeroDivisionError:
                continue

        try:
            return total / self.cm.total_weight
        except ZeroDivisionError:
            return 0.0


class MultiFBeta(metrics.base.MultiClassMetric):
    """Multi-class F-Beta score with different betas per class.

    The multiclass F-Beta score is the arithmetic average of the binary F-Beta scores of each class.
    The mean can be weighted by providing class weights.

    Parameters
    ----------
    betas
        Weight of precision in the harmonic mean of each class.
    weights
        Class weights. If not provided then uniform weights will be used.
    cm
        This parameter allows sharing the same confusion
        matrix between multiple metrics. Sharing a confusion matrix reduces the amount of storage
        and computation time.

    Examples
    --------

    >>> from river import metrics

    >>> y_true = [0, 1, 2, 2, 2]
    >>> y_pred = [0, 0, 2, 2, 1]

    >>> metric = metrics.MultiFBeta(
    ...     betas={0: 0.25, 1: 1, 2: 4},
    ...     weights={0: 1, 1: 1, 2: 2}
    ... )

    >>> for yt, yp in zip(y_true, y_pred):
    ...     print(metric.update(yt, yp))
    MultiFBeta: 100.00%
    MultiFBeta: 25.76%
    MultiFBeta: 62.88%
    MultiFBeta: 62.88%
    MultiFBeta: 46.88%

    """

    def __init__(self, betas, weights, cm=None):
        super().__init__(cm)
        self.betas = betas
        self.weights = (
            collections.defaultdict(functools.partial(int, 1)) if weights is None else weights
        )

    def get(self):
        total = 0

        for c in self.cm.classes:

            b2 = self.betas[c] ** 2

            try:
                p = self.cm[c][c] / self.cm.sum_col[c]
            except ZeroDivisionError:
                p = 0

            try:
                r = self.cm[c][c] / self.cm.sum_row[c]
            except ZeroDivisionError:
                r = 0

            try:
                total += self.weights[c] * (1 + b2) * p * r / (b2 * p + r)
            except ZeroDivisionError:
                continue

        try:
            return total / sum(self.weights[c] for c in self.cm.classes)
        except ZeroDivisionError:
            return 0.0


class F1(FBeta):
    """Binary F1 score.

    Parameters
    ----------
    cm
        This parameter allows sharing the same confusion
        matrix between multiple metrics. Sharing a confusion matrix reduces the amount of storage
        and computation time.
    pos_val
        Value to treat as "positive".

    Examples
    --------

    >>> from river import metrics

    >>> y_true = [False, False, False, True, True, True]
    >>> y_pred = [False, False, True, True, False, False]

    >>> metric = metrics.F1()

    >>> for yt, yp in zip(y_true, y_pred):
    ...     metric = metric.update(yt, yp)

    >>> metric
    F1: 40.00%

    """

    def __init__(self, cm=None, pos_val=True):
        super().__init__(beta=1.0, cm=cm, pos_val=pos_val)


class MacroF1(MacroFBeta):
    """Macro-average F1 score.

    This works by computing the F1 score per class, and then performs a global average.

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

    >>> metric = metrics.MacroF1()

    >>> for yt, yp in zip(y_true, y_pred):
    ...     print(metric.update(yt, yp))
    MacroF1: 100.00%
    MacroF1: 33.33%
    MacroF1: 55.56%
    MacroF1: 55.56%
    MacroF1: 48.89%

    """

    def __init__(self, cm=None):
        super().__init__(beta=1.0, cm=cm)


class MicroF1(MicroFBeta):
    """Micro-average F1 score.

    This computes the F1 score by merging all the predictions and true labels, and then computes a
    global F1 score.

    Parameters
    ----------
    cm
        This parameter allows sharing the same confusion
        matrix between multiple metrics. Sharing a confusion matrix reduces the amount of storage
        and computation time.

    Examples
    --------

    >>> from river import metrics

    >>> y_true = [0, 1, 2, 2, 0]
    >>> y_pred = [0, 1, 1, 2, 1]

    >>> metric = metrics.MicroF1()
    >>> for yt, yp in zip(y_true, y_pred):
    ...     metric = metric.update(yt, yp)

    >>> metric
    MicroF1: 60.00%

    References
    ----------
    [^1]: [Why are precision, recall and F1 score equal when using micro averaging in a multi-class problem?](https://simonhessner.de/why-are-precision-recall-and-f1-score-equal-when-using-micro-averaging-in-a-multi-class-problem/)

    """

    def __init__(self, cm=None):
        super().__init__(beta=1.0, cm=cm)


class WeightedF1(WeightedFBeta):
    """Weighted-average F1 score.

    This works by computing the F1 score per class, and then performs a global weighted average by
    using the support of each class.

    Parameters
    ----------
    cm
        This parameter allows sharing the same confusion
        matrix between multiple metrics. Sharing a confusion matrix reduces the amount of storage
        and computation time.
    pos_val
        Value to treat as "positive".

    Examples
    --------

    >>> from river import metrics

    >>> y_true = [0, 1, 2, 2, 2]
    >>> y_pred = [0, 0, 2, 2, 1]

    >>> metric = metrics.WeightedF1()

    >>> for yt, yp in zip(y_true, y_pred):
    ...     print(metric.update(yt, yp))
    WeightedF1: 100.00%
    WeightedF1: 33.33%
    WeightedF1: 55.56%
    WeightedF1: 66.67%
    WeightedF1: 61.33%

    """

    def __init__(self, cm=None):
        super().__init__(beta=1.0, cm=cm)
