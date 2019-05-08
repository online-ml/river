from .. import stats

from . import base


__all__ = ['Accuracy']


class Accuracy(stats.Mean, base.MultiClassificationMetric):
    """Accuracy score, which is the percentage of exact matches.

    Example:

        ::

            >>> import math
            >>> from creme import metrics
            >>> from sklearn.metrics import accuracy_score

            >>> y_true = [True, False, True, True, True]
            >>> y_pred = [True, True, False, True, True]

            >>> metric = metrics.Accuracy()
            >>> for i, (y_t, y_p) in enumerate(zip(y_true, y_pred)):
            ...     metric = metric.update(y_t, y_p)
            ...     assert math.isclose(metric.get(), accuracy_score(y_true[:i+1], y_pred[:i+1]))

            >>> metric
            Accuracy: 0.6

    """

    @property
    def bigger_is_better(self):
        return True

    @property
    def requires_labels(self):
        return True

    def update(self, y_true, y_pred):
        return super().update(y_true == y_pred)
