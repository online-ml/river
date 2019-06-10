from . import fbeta
from . import precision


__all__ = [
    'F1',
    'MicroF1',
    'MultiF1',
    'RollingF1',
    'RollingMicroF1',
    'RollingMultiF1'
]


class F1(fbeta.FBeta):
    """Binary F1 score.

    The F1 score is the harmonic mean of the precision and the recall.

    Example:

        ::

            >>> from creme import metrics

            >>> y_true = [True, False, True, True, True]
            >>> y_pred = [True, True, False, True, True]

            >>> metric = metrics.F1()

            >>> for yt, yp in zip(y_true, y_pred):
            ...     print(metric.update(yt, yp))
            F1: 1.
            F1: 0.666667
            F1: 0.5
            F1: 0.666667
            F1: 0.75

    """

    def __init__(self):
        super().__init__(beta=1)


class RollingF1(fbeta.RollingFBeta):
    """Rolling binary F1 score.

    The F1 score is the harmonic mean of the precision and the recall.

    Example:

        ::

            >>> from creme import metrics

            >>> y_true = [True, False, True, True, True]
            >>> y_pred = [True, True, False, True, True]

            >>> metric = metrics.RollingF1(window_size=3)

            >>> for yt, yp in zip(y_true, y_pred):
            ...     print(metric.update(yt, yp))
            RollingF1: 1.
            RollingF1: 0.666667
            RollingF1: 0.5
            RollingF1: 0.5
            RollingF1: 0.8

    """

    def __init__(self, window_size):
        super().__init__(beta=1, window_size=window_size)


class MultiF1(fbeta.MultiFBeta):
    """Multiclass F1 score.

    The multiclass F1 score is the arithmetic average of the binary F1 scores of each class.
    The mean can be weighted by providing class weights.

    Parameters:
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

            >>> metric = metrics.MultiF1(weights={2: 2})

            >>> for yt, yp in zip(y_true, y_pred):
            ...     print(metric.update(yt, yp))
            MultiF1: 1.
            MultiF1: 0.333333
            MultiF1: 0.666667
            MultiF1: 0.666667
            MultiF1: 0.566667

    """

    def __init__(self, default_weight=1, weights=None):
        super().__init__(beta=1, default_weight=default_weight, weights=weights)


class RollingMultiF1(fbeta.RollingMultiFBeta):
    """Rolling multiclass F1 score.

    The multiclass F1 score is the arithmetic average of the binary F1 scores of each class.
    The mean can be weighted by providing class weights.

    Parameters:
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

            >>> metric = metrics.RollingMultiF1(window_size=4)

            >>> for yt, yp in zip(y_true, y_pred):
            ...     print(metric.update(yt, yp))
            RollingMultiF1: 1.
            RollingMultiF1: 0.333333
            RollingMultiF1: 0.555556
            RollingMultiF1: 0.555556
            RollingMultiF1: 0.333333
            RollingMultiF1: 0.555556

    """

    def __init__(self, window_size, default_weight=1, weights=None):
        super().__init__(
            beta=1,
            default_weight=default_weight,
            weights=weights,
            window_size=window_size
        )


class MicroF1(precision.MicroPrecision):
    """Micro-average F1 score.

    The micro-average F1 score is exactly equivalent to the micro-average precision as well as the
    micro-average recall score.

    Example:

        ::

            >>> from creme import metrics

            >>> y_true = [0, 1, 2, 2, 2]
            >>> y_pred = [0, 0, 2, 2, 1]

            >>> metric = metrics.MicroF1()

            >>> for yt, yp in zip(y_true, y_pred):
            ...     print(metric.update(yt, yp))
            MicroF1: 1.
            MicroF1: 0.5
            MicroF1: 0.666667
            MicroF1: 0.75
            MicroF1: 0.6

    References:
        1. `Why are precision, recall and F1 score equal when using micro averaging in a multi-class problem? <https://simonhessner.de/why-are-precision-recall-and-f1-score-equal-when-using-micro-averaging-in-a-multi-class-problem/>`_

    """


class RollingMicroF1(precision.RollingMicroPrecision):
    """Rolling micro-average F1 score.

    The micro-average F1 score is exactly equivalent to the micro-average precision as well as the
    micro-average recall score.

    Example:

        ::

            >>> from creme import metrics

            >>> y_true = [0, 1, 2, 2, 2]
            >>> y_pred = [0, 0, 2, 2, 1]

            >>> metric = metrics.RollingMicroF1(window_size=3)

            >>> for yt, yp in zip(y_true, y_pred):
            ...     print(metric.update(yt, yp).get())
            1.0
            0.5
            0.666666...
            0.666666...
            0.666666...

            >>> metric
            RollingMicroF1: 0.666667

    References:
        1. `Why are precision, recall and F1 score equal when using micro averaging in a multi-class problem? <https://simonhessner.de/why-are-precision-recall-and-f1-score-equal-when-using-micro-averaging-in-a-multi-class-problem/>`_

    """
