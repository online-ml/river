from .. import optim
from .. import stats

from . import base


__all__ = ['CrossEntropy']


class CrossEntropy(stats.Mean, base.MultiClassificationMetric):
    """Multiclass generalization of the logarithmic loss.

    Example:

        ::

            >>> from creme import metrics
            >>> from sklearn.metrics import log_loss

            >>> y_true = [0, 1, 2, 2]
            >>> y_pred = [
            ...     [0.29450637, 0.34216758, 0.36332605],
            ...     [0.21290077, 0.32728332, 0.45981591],
            ...     [0.42860913, 0.33380113, 0.23758974],
            ...     [0.44941979, 0.32962558, 0.22095463]
            ... ]

            >>> metric = metrics.CrossEntropy()
            >>> labels = [0, 1, 2]

            >>> for i, (y_t, y_p) in enumerate(zip(y_true, y_pred)):
            ...     metric = metric.update(y_t, {i: p for i, p in enumerate(y_p)})
            ...     if i >= 1:
            ...         assert metric.get() == log_loss(y_true[:i+1], y_pred[:i+1], labels=labels)

            >>> metric
            CrossEntropy: 1.321598

    """

    @property
    def bigger_is_better(self):
        return False

    @property
    def requires_labels(self):
        return False

    def update(self, y_true, y_pred):
        ce = optim.CrossEntropy().__call__
        return super().update(ce(y_true, y_pred))
