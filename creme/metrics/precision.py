from .. import stats

from . import base


class Precision(stats.Mean, base.BinaryClassificationMetric):
    """

    Example:

        >>> from creme import metrics

        >>> y_true = [True, False, True, True, True]
        >>> y_pred = [True, True, False, True, True]

        >>> metric = metrics.Precision()
        >>> for y_t, y_p in zip(y_true, y_pred):
        ...     print(metric.update(y_t, y_p).get())
        1.0
        1.0
        0.5
        0.666666...
        0.75

        >>> metric
        Precision: 0.75

    """

    @property
    def requires_labels(self):
        return True

    def update(self, y_true, y_pred):
        if y_true:
            return super().update(y_true == y_pred)
        return self
