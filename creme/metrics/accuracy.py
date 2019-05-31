from .. import stats

from . import base


__all__ = ['Accuracy', 'RollingAccuracy']


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

    def update(self, y_true, y_pred):
        return super().update(y_true == y_pred)


class RollingAccuracy(stats.RollingMean, BaseAccuracy):
    """Rolling accuracy score, which is the percentage of exact matches over a window.

    Example:

        ::

            >>> from creme import metrics

            >>> y_true = [True, False, True, True, True]
            >>> y_pred = [True, True, False, True, True]

            >>> metric = metrics.RollingAccuracy(window_size=3)
            >>> for y_t, y_p in zip(y_true, y_pred):
            ...     print(metric.update(y_t, y_p))
            RollingAccuracy: 1.
            RollingAccuracy: 0.5
            RollingAccuracy: 0.333333
            RollingAccuracy: 0.333333
            RollingAccuracy: 0.666667

    """

    def update(self, y_true, y_pred):
        return super().update(y_true == y_pred)
