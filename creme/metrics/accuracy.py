from .. import stats

from . import base


__all__ = ['Accuracy']


class BaseAccuracy(base.MultiClassMetric):

    @property
    def bigger_is_better(self):
        return True

    @property
    def requires_labels(self):
        return True


class Accuracy(stats.Mean, BaseAccuracy):
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

    def update(self, y_true, y_pred, sample_weight=1.):
        return super().update(x=y_true == y_pred, w=sample_weight)

    def revert(self, y_true, y_pred, sample_weight=1.):
        return super().revert(x=y_true == y_pred, w=sample_weight)
