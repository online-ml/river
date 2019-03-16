from .. import stats

from . import base


class Accuracy(stats.Mean, base.MultiClassificationMetric):
    """Accuracy score, which is the ratio of exact matches.

    Example:

    ::

        >>> from creme import metrics

        >>> y_true = [0, 1, 2, 3]
        >>> y_pred = [0, 2, 1, 3]

        >>> metric = metrics.Accuracy()
        >>> for y_t, y_p in zip(y_true, y_pred):
        ...     print(metric.update(y_t, y_p).get())
        1.0
        0.5
        0.333333...
        0.5

        >>> metric
        Accuracy: 0.5

    """

    def requires_labels(self):
        return True

    def update(self, y_true, y_pred):
        return super().update(y_true == y_pred)
