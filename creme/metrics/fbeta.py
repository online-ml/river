import collections
import functools

from . import base
from . import precision
from . import recall


__all__ = [
    'F1',
    'FBeta',
    'MacroF1',
    'MacroFBeta',
    'MicroF1',
    'MicroFBeta',
    'MultiFBeta',
    'WeightedF1',
    'WeightedFBeta'
]


class BaseFBeta:

    @property
    def bigger_is_better(self):
        return True

    @property
    def requires_labels(self):
        return True


class FBeta(BaseFBeta, base.BinaryMetric):
    """Binary F-Beta score.

    The FBeta score is a weighted harmonic mean between precision and recall. The higher the
    ``beta`` value, the higher the recall will be taken into account. When ``beta`` equals 1,
    precision and recall and equivalently weighted, which results in the F1 score (see
    `metrics.F1`).

    Parameters:
        beta (float): Weight of precision in harmonic mean.

    Attributes:
        precision (metrics.Precision): Precision score.
        recall (metrics.Recall): Recall score.

    Example:

        ::

            >>> from creme import metrics

            >>> y_true = [False, False, False, True, True, True]
            >>> y_pred = [False, False, True, True, False, False]

            >>> metric = metrics.FBeta(beta=2)
            >>> for y_t, y_p in zip(y_true, y_pred):
            ...     metric = metric.update(y_t, y_p)

            >>> metric
            FBeta: 0.357143

    """

    def __init__(self, beta):

        if beta <= 0:
            raise ValueError('beta should be strictly positive')

        self.beta = beta
        self.precision = precision.Precision()
        self.recall = recall.Recall()

    def update(self, y_true, y_pred, sample_weight=1.):
        self.precision.update(y_true, y_pred, sample_weight)
        self.recall.update(y_true, y_pred, sample_weight)
        return self

    def revert(self, y_true, y_pred, sample_weight=1.):
        self.precision.revert(y_true, y_pred, sample_weight)
        self.recall.revert(y_true, y_pred, sample_weight)
        return self

    def get(self):
        p = self.precision.get()
        r = self.recall.get()
        b2 = self.beta ** 2
        try:
            return (1 + b2) * p * r / (b2 * p + r)
        except ZeroDivisionError:
            return 0.


class MacroFBeta(BaseFBeta, base.MultiClassMetric):
    """Macro-average F-Beta score.

    This works by computing the F-Beta score per class, and then performs a global average.

    Parameters:
        beta (float): Weight of precision in harmonic mean.

    Attributes:
        fbetas (collections.defaultdict): F-Beta scores per class.

    Example:

        ::

            >>> from creme import metrics

            >>> y_true = [0, 1, 2, 2, 2]
            >>> y_pred = [0, 0, 2, 2, 1]

            >>> metric = metrics.MacroFBeta(beta=0.8)

            >>> for yt, yp in zip(y_true, y_pred):
            ...     print(metric.update(yt, yp))
            MacroFBeta: 1.
            MacroFBeta: 0.310606
            MacroFBeta: 0.540404
            MacroFBeta: 0.540404
            MacroFBeta: 0.485982

    """

    def __init__(self, beta):
        self.fbetas = collections.defaultdict(functools.partial(FBeta, beta=beta))
        self._class_counts = collections.Counter()

    def update(self, y_true, y_pred, sample_weight=1.):
        self._class_counts.update([y_true, y_pred])

        for c in self._class_counts:
            self.fbetas[c].update(y_true == c, y_pred == c, sample_weight)

        return self

    def revert(self, y_true, y_pred, sample_weight=1.):
        self._class_counts.subtract([y_true, y_pred])

        for c in self._class_counts:
            self.fbetas[c].revert(y_true == c, y_pred == c, sample_weight)

        return self

    def get(self):
        relevant = [c for c, count in self._class_counts.items() if count > 0]
        try:
            return sum(self.fbetas[c].get() for c in relevant) / len(relevant)
        except ZeroDivisionError:
            return 0.


class MicroFBeta(FBeta, base.MultiClassMetric):
    """Micro-average F-Beta score.

    This computes the F-Beta score by merging all the predictions and true labels, and then
    computes a global F-Beta score.

    Parameters:
        beta (float): Weight of precision in harmonic mean.

    Attributes:
        precision (metrics.Precision): Precision score.
        recall (metrics.Recall): Recall score.

    Example:

        ::

            >>> from creme import metrics

            >>> y_true = [0, 1, 2, 2, 0]
            >>> y_pred = [0, 1, 1, 2, 1]

            >>> metric = metrics.MicroFBeta(beta=2)
            >>> for y_t, y_p in zip(y_true, y_pred):
            ...     metric = metric.update(y_t, y_p)

            >>> metric
            MicroFBeta: 0.6

    References:
        1. `Why are precision, recall and F1 score equal when using micro averaging in a multi-class problem? <https://simonhessner.de/why-are-precision-recall-and-f1-score-equal-when-using-micro-averaging-in-a-multi-class-problem/>`_

    """

    def __init__(self, beta):
        super().__init__(beta=beta)
        self.precision = precision.MicroPrecision()
        self.recall = recall.MicroRecall()


class WeightedFBeta(BaseFBeta, base.MultiClassMetric):
    """Weighted-average F-Beta score.

    This works by computing the F-Beta score per class, and then performs a global weighted average
    according to the support of each class.

    Parameters:
        beta (float): Weight of precision in harmonic mean.

    Attributes:
        fbetas (collections.defaultdict): F-Beta scores per class.

    Example:

        ::

            >>> from creme import metrics

            >>> y_true = [0, 1, 2, 2, 2]
            >>> y_pred = [0, 0, 2, 2, 1]

            >>> metric = metrics.WeightedFBeta(beta=0.8)

            >>> for yt, yp in zip(y_true, y_pred):
            ...     print(metric.update(yt, yp))
            WeightedFBeta: 1.
            WeightedFBeta: 0.310606
            WeightedFBeta: 0.540404
            WeightedFBeta: 0.655303
            WeightedFBeta: 0.626283

    """

    def __init__(self, beta):
        self.fbetas = collections.defaultdict(functools.partial(FBeta, beta=beta))
        self.support = collections.Counter()
        self._class_counts = collections.Counter()

    def update(self, y_true, y_pred, sample_weight=1.):
        self._class_counts.update([y_true, y_pred])
        self.support.update({y_true: sample_weight})

        for c in self._class_counts:
            self.fbetas[c].update(y_true == c, y_pred == c, sample_weight)

        return self

    def revert(self, y_true, y_pred, sample_weight=1.):
        self._class_counts.subtract([y_true, y_pred])
        self.support.subtract({y_true: sample_weight})

        for c in self._class_counts:
            self.fbetas[c].revert(y_true == c, y_pred == c, sample_weight)

        return self

    def get(self):
        relevant = [c for c, count in self._class_counts.items() if count > 0]
        try:
            return (
                sum(self.fbetas[c].get() * self.support[c] for c in relevant) /
                sum(self.support[c] for c in relevant)
            )
        except ZeroDivisionError:
            return 0.


class MultiFBeta(BaseFBeta, base.MultiClassMetric):
    """Multi-class F-Beta score with different betas per class.

    The multiclass F-Beta score is the arithmetic average of the binary F-Beta scores of each class.
    The mean can be weighted by providing class weights.

    Parameters:
        beta (dict): Weight of precision in harmonic mean. For different beta values per class,
            provide a dict with class labels as keys and beta values as values.
        weights (dict): Class weights. If not provided then uniform weights will be used

    Attributes:
        fbetas (collections.defaultdict): Classes fbeta values.

    Example:

        ::

            >>> from creme import metrics

            >>> y_true = [0, 1, 2, 2, 2]
            >>> y_pred = [0, 0, 2, 2, 1]

            >>> metric = metrics.MultiFBeta(
            ...     betas={0: 0.25, 1: 1, 2: 4},
            ...     weights={0: 1, 1: 1, 2: 2}
            ... )

            >>> for yt, yp in zip(y_true, y_pred):
            ...     print(metric.update(yt, yp))
            MultiFBeta: 1.
            MultiFBeta: 0.257576
            MultiFBeta: 0.628788
            MultiFBeta: 0.628788
            MultiFBeta: 0.468788

    """

    def __init__(self, betas, weights=None):
        self.betas = betas
        self.fbetas = dict()
        self._class_counts = collections.Counter()
        self.weights = (
            weights
            if weights is not None
            else collections.defaultdict(functools.partial(int, 1))
        )

    def update(self, y_true, y_pred, sample_weight=1.):
        self._class_counts.update([y_true, y_pred])

        for c in self._class_counts:
            try:
                fb = self.fbetas[c]
            except KeyError:
                fb = self.fbetas[c] = FBeta(beta=self.betas[c])

            fb.update(y_true=y_true == c, y_pred=y_pred == c, sample_weight=sample_weight)

        return self

    def revert(self, y_true, y_pred, sample_weight=1.):
        self._class_counts.subtract([y_true, y_pred])

        for c in self._class_counts:
            try:
                fb = self.fbetas[c]
            except KeyError:
                fb = self.fbetas[c] = FBeta(beta=self.betas[c])

            fb.revert(y_true=y_true == c, y_pred=y_pred == c, sample_weight=sample_weight)

        return self

    def get(self):
        relevant = [c for c, count in self._class_counts.items() if count > 0]
        try:
            return (
                sum(self.fbetas[c].get() * self.weights[c] for c in relevant) /
                sum(self.weights[c] for c in relevant)
            )
        except ZeroDivisionError:
            return 0.


class F1(FBeta):
    """Binary F1 score.

    Attributes:
        precision (metrics.Precision): Precision score.
        recall (metrics.Recall): Recall score.

    Example:

        ::

            >>> from creme import metrics

            >>> y_true = [False, False, False, True, True, True]
            >>> y_pred = [False, False, True, True, False, False]

            >>> metric = metrics.F1()
            >>> for y_t, y_p in zip(y_true, y_pred):
            ...     metric = metric.update(y_t, y_p)

            >>> metric
            F1: 0.4

    """

    def __init__(self):
        super().__init__(beta=1.)


class MacroF1(MacroFBeta):
    """Macro-average F1 score.

    This works by computing the F1 score per class, and then performs a global average.

    Attributes:
        fbetas (collections.defaultdict): F-Beta scores per class.

    Example:

        ::

            >>> from creme import metrics

            >>> y_true = [0, 1, 2, 2, 2]
            >>> y_pred = [0, 0, 2, 2, 1]

            >>> metric = metrics.MacroF1()

            >>> for yt, yp in zip(y_true, y_pred):
            ...     print(metric.update(yt, yp))
            MacroF1: 1.
            MacroF1: 0.333333
            MacroF1: 0.555556
            MacroF1: 0.555556
            MacroF1: 0.488889

    """

    def __init__(self):
        super().__init__(beta=1.)


class MicroF1(MicroFBeta):
    """Micro-average F1 score.

    This computes the F1 score by merging all the predictions and true labels, and then computes a
    global F1 score.

    Attributes:
        precision (metrics.Precision): Precision score.
        recall (metrics.Recall): Recall score.

    Example:

        ::

            >>> from creme import metrics

            >>> y_true = [0, 1, 2, 2, 0]
            >>> y_pred = [0, 1, 1, 2, 1]

            >>> metric = metrics.MicroF1()
            >>> for y_t, y_p in zip(y_true, y_pred):
            ...     metric = metric.update(y_t, y_p)

            >>> metric
            MicroF1: 0.6

    References:
        1. `Why are precision, recall and F1 score equal when using micro averaging in a multi-class problem? <https://simonhessner.de/why-are-precision-recall-and-f1-score-equal-when-using-micro-averaging-in-a-multi-class-problem/>`_

    """

    def __init__(self):
        super().__init__(beta=1.)


class WeightedF1(WeightedFBeta):
    """Weighted-average F1 score.

    This works by computing the F1 score per class, and then performs a global weighted average by
    using the support of each class.

    Attributes:
        fbetas (collections.defaultdict): F-Beta scores per class.

    Example:

        ::

            >>> from creme import metrics

            >>> y_true = [0, 1, 2, 2, 2]
            >>> y_pred = [0, 0, 2, 2, 1]

            >>> metric = metrics.WeightedF1()

            >>> for yt, yp in zip(y_true, y_pred):
            ...     print(metric.update(yt, yp))
            WeightedF1: 1.
            WeightedF1: 0.333333
            WeightedF1: 0.555556
            WeightedF1: 0.666667
            WeightedF1: 0.613333

    """

    def __init__(self):
        super().__init__(beta=1.)
