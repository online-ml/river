import collections

from . import base
from . import precision
from . import recall


__all__ = [
    'FBeta',
    'MicroFBeta',
    'MultiFBeta',
    'RollingFBeta',
    'RollingMicroFBeta',
    'RollingMultiFBeta'
]


class BaseFBeta:

    @property
    def bigger_is_better(self):
        return True

    @property
    def requires_labels(self):
        return True


class FBeta(BaseFBeta, base.BinaryMetric):
    """Binary FBeta score.

    The FBeta score is a measure of a test accuracy considering both precision and recall. It is a
    weighted harmonic mean bounded between 0 and 1.
    `beta` parameter allows to weight recall higher than precision if sets to more than 1 or lower
    if sets between 0 and 1. For `beta` equals to 1, precision and recall and equivalently weighted
    resulting to traditionnal F1 score (see `metrics.F1Score`).

    Parameters:
        beta (float): Weight of precision in harmonic mean.

    Attributes:
        beta2 (float): Beta squared value.
        precision (metric.Precision): Precision score.
        recall (metric.Recall): Recall score.

    Example:

        ::

            >>> from creme import metrics

            >>> y_true = [False, False, False, True, True, True]
            >>> y_pred = [False, False, True, True, False, False]

            >>> metric = metrics.FBeta(beta=2)
            >>> for i, (y_t, y_p) in enumerate(zip(y_true, y_pred)):
            ...     metric = metric.update(y_t, y_p)

            >>> metric
            FBeta: 0.357143

    """

    def __init__(self, beta):
        self.precision = precision.Precision()
        self.recall = recall.Recall()

        if beta <= 0:
            raise ValueError('beta should be higher than 0!')

        self.beta2 = beta ** 2

    def update(self, y_true, y_pred):
        self.precision.update(y_true, y_pred)
        self.recall.update(y_true, y_pred)
        return self

    def get(self):
        prec, recall, beta2 = self.precision.get(), self.recall.get(), self.beta2
        try:
            return (1 + beta2) * prec * recall / (beta2 * prec + recall)
        except ZeroDivisionError:
            return 0.0


class RollingFBeta(FBeta):
    """Rolling binary FBeta score.

    The FBeta score is a measure of a test accuracy considering both precision and recall. It is a
    weighted harmonic mean bounded between 0 and 1.
    `beta` parameter allows to weight recall higher than precision if sets to more than 1 or lower
    if sets between 0 and 1. For `beta` equals to 1, precision and recall and equivalently weighted
    resulting to traditionnal F1 score (see `metrics.F1Score`).

    Parameters:
        beta (float): Weight of precision in harmonic mean.
        window_size (int): Size of the rolling window.

    Attributes:
        beta2 (float): Beta squared value.
        precision (metric.RollingPrecision): Rolling precision score.
        recall (metric.RollingRecall): Rolling recall score.

    Example:

        ::

            >>> from creme import metrics

            >>> y_true = [False, False, False, True, True, True]
            >>> y_pred = [False, False, True, True, False, False]

            >>> metric = metrics.RollingFBeta(beta=2, window_size=3)
            >>> for y_t, y_p in zip(y_true, y_pred):
            ...     print(metric.update(y_t, y_p).get())
            0.0
            0.0
            0.0
            0.8333333333333334
            0.5
            0.3846153846153846

    """

    def __init__(self, beta, window_size):
        super().__init__(beta)
        self.precision = precision.RollingPrecision(window_size=window_size)
        self.recall = recall.RollingRecall(window_size=window_size)
        self.window_size = window_size


class MultiFBeta(BaseFBeta, base.MultiClassMetric):
    """Multiclass FBeta score.

    The Multiclass FBeta score is the arithmetic average of the binary FBeta scores of each class.
    The mean can be weighted by providing class weights.

    Parameters:
        beta (float or dict): Weight of precision in harmonic mean. For different beta values per
            class, provide a dict with class labels as keys and beta values as values. For classes
            not specified, beta will be set to `default_beta`.
        default_beta (float): Used when `beta` is a dict. Default beta value to assign to unknown
            classes. Defaults to 1.
        default_weight (float): Default class weight. Defaults to 1.
        weights (dict of floats): Class weights. Defaults to `None`.

    Attributes:
        classes (set): Metric different classes.
        fbeta_s (collections.defaultdict): Classes fbeta values.

    Example:

        ::

            >>> from creme import metrics

            >>> y_true = [0, 1, 2, 2, 2]
            >>> y_pred = [0, 0, 2, 2, 1]

            >>> metric = metrics.MultiFBeta(beta={0: 0.25, 1: 1, 2: 4}, weights={2: 2})

            >>> for yt, yp in zip(y_true, y_pred):
            ...     print(metric.update(yt, yp))
            MultiFBeta: 1.
            MultiFBeta: 0.257576
            MultiFBeta: 0.628788
            MultiFBeta: 0.628788
            MultiFBeta: 0.468788

    """

    def __init__(self, beta, default_beta=1, default_weight=1, weights=None):
        if isinstance(beta, dict):
            self.beta = collections.defaultdict(lambda: default_beta, beta)
        else:
            self.beta = collections.defaultdict(lambda: beta)

        self.fbeta_s = dict()
        self.classes = set()

        if weights:
            self.weights = collections.defaultdict(lambda: default_weight, weights)
        else:
            self.weights = collections.defaultdict(lambda: 1)

    def update(self, y_true, y_pred):
        self.classes.update({y_true, y_pred})
        for c in self.classes:
            try:
                self.fbeta_s[c].update(y_true == c, y_pred == c)
            except KeyError:
                self.fbeta_s[c] = FBeta(self.beta[c]).update(y_true == c, y_pred == c)
        return self

    def get(self):
        total = sum(self.fbeta_s[label].get() * self.weights[label] for label in self.fbeta_s.keys())
        denom = sum(self.weights[label] for label in self.fbeta_s.keys())
        try:
            return total / denom
        except ZeroDivisionError:
            return 0.


class RollingMultiFBeta(MultiFBeta):
    """Rolling multiclass FBeta score.

    The Multiclass FBeta score is the arithmetic average of the binary FBeta scores of each class.
    The mean can be weighted by providing class weights.

    Parameters:
        beta (float or dict): Weight of precision in harmonic mean. For different beta values per
            class, provide a dict with class labels as keys and beta values as values. For classes
            not specified, beta will be set to `default_beta`.
        default_beta (float): Used when `beta` is a dict. Default beta value to assign to unknown
            classes. Defaults to 1.
        default_weight (float): Default class weight. Defaults to 1.
        weights (dict of floats): Class weights. Defaults to `None`.
        window_size (int): Size of the rolling window.

    Attributes:
        classes (set): Metric different classes.
        fbeta_s (collections.defaultdict): Classes fbeta values.

    Example:

        ::

            >>> from creme import metrics

            >>> y_true = [0, 1, 2, 2, 0, 1]
            >>> y_pred = [0, 0, 2, 2, 1, 1]

            >>> metric = metrics.RollingMultiFBeta(beta=1, window_size=4)

            >>> for yt, yp in zip(y_true, y_pred):
            ...     print(metric.update(yt, yp))
            RollingMultiFBeta: 1.
            RollingMultiFBeta: 0.333333
            RollingMultiFBeta: 0.555556
            RollingMultiFBeta: 0.555556
            RollingMultiFBeta: 0.333333
            RollingMultiFBeta: 0.555556

    """

    def __init__(self, beta, window_size, default_beta=1, default_weight=1, weights=None):
        super().__init__(
            beta=beta,
            default_beta=default_beta,
            default_weight=default_weight,
            weights=weights
        )
        self.window_size = window_size

    def update(self, y_true, y_pred):
        self.classes.update({y_true, y_pred})
        for c in self.classes:
            try:
                self.fbeta_s[c].update(y_true == c, y_pred == c)
            except KeyError:
                self.fbeta_s[c] = RollingFBeta(self.beta[c], self.window_size)
                self.fbeta_s[c].update(y_true == c, y_pred == c)
        return self


class MicroFBeta(FBeta, base.MultiClassMetric):
    """Micro multiclass FBeta score.

    The Micro FBeta score is a specific case of the FBeta score where the metric is globally
    computed by counting the total true positives, false negatives and false positives.
    `beta` parameter allows to weight recall higher than precision if sets to more than 1 or lower
    if sets between 0 and 1. For `beta` equals to 1, precision and recall and equivalently weighted
    resulting to traditional micro F1 score (see `metrics.MicroF1`).

    Parameters:
        beta (float): Weight of precision in harmonic mean.

    Attributes:
        beta2 (float): Beta squared value.
        precision (metric.Precision): Precision score.
        recall (metric.Recall): Recall score.

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

    """

    def __init__(self, beta):
        super().__init__(beta)
        self.precision = precision.MicroPrecision()
        self.recall = recall.MicroRecall()


class RollingMicroFBeta(RollingFBeta, base.MultiClassMetric):
    """Rolling micro multiclass FBeta score.

    The Micro FBeta score is a specific case of the FBeta score where the metric is globally
    computed by counting the total true positives, false negatives and false positives.
    `beta` parameter allows to weight recall higher than precision if sets to more than 1 or lower
    if sets between 0 and 1. For `beta` equals to 1, precision and recall and equivalently weighted
    resulting to traditional micro F1 score (see `metrics.MicroF1`).

    Parameters:
        beta (float): Weight of precision in harmonic mean.
        window_size (int): Size of the rolling window.

    Attributes:
        beta2 (float): Beta squared value.
        precision (metric.RollingPrecision): Rolling precision score.
        recall (metric.RollingRecall): Rolling recall score.

    Example:

        ::

            >>> from creme import metrics

            >>> y_true = [0, 1, 2, 2, 0]
            >>> y_pred = [0, 1, 1, 2, 1]

            >>> metric = metrics.RollingMicroFBeta(beta=2, window_size=3)
            >>> for y_t, y_p in zip(y_true, y_pred):
            ...     print(metric.update(y_t, y_p).get())
            1.0
            1.0
            0.6666666666666666
            0.6666666666666666
            0.3333333333333333

    """

    def __init__(self, beta, window_size):
        super().__init__(beta, window_size)
        self.precision = precision.RollingMicroPrecision(window_size=window_size)
        self.recall = recall.RollingMicroRecall(window_size=window_size)
