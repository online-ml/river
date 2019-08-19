from .. import optim
from .. import stats

from . import base


__all__ = ['CrossEntropy', 'RollingCrossEntropy']


class BaseCrossEntropy(base.MultiClassMetric):

    @property
    def bigger_is_better(self):
        return False

    @property
    def requires_labels(self):
        return False


class CrossEntropy(stats.Mean, BaseCrossEntropy):
    """Multiclass generalization of the logarithmic loss.

    Example:

        ::

            >>> from creme import metrics
            >>> from sklearn.metrics import log_loss

            >>> y_true = [0, 1, 2, 2]
            >>> y_pred = [
            ...     {0: 0.29450637, 1: 0.34216758, 2: 0.36332605},
            ...     {0: 0.21290077, 1: 0.32728332, 2: 0.45981591},
            ...     {0: 0.42860913, 1: 0.33380113, 2: 0.23758974},
            ...     {0: 0.44941979, 1: 0.32962558, 2: 0.22095463}
            ... ]

            >>> metric = metrics.CrossEntropy()

            >>> for y_t, y_p in zip(y_true, y_pred):
            ...     metric = metric.update(y_t, y_p)
            ...     print(metric.get())
            1.222454...
            1.169691...
            1.258864...
            1.321597...

            >>> metric
            CrossEntropy: 1.321598

    """

    def update(self, y_true, y_pred):
        return super().update(optim.CrossEntropy().eval(y_true, y_pred))


class RollingCrossEntropy(stats.RollingMean, BaseCrossEntropy):
    """Rolling cross-entropy.

    Parameters:
        window_size (int): Size of the window of recent values to consider.

    Example:

        ::

            >>> from creme import metrics
            >>> from sklearn.metrics import log_loss

            >>> y_true = [0, 1, 2, 2]
            >>> y_pred = [
            ...     {0: 0.29450637, 1: 0.34216758, 2: 0.36332605},
            ...     {0: 0.21290077, 1: 0.32728332, 2: 0.45981591},
            ...     {0: 0.42860913, 1: 0.33380113, 2: 0.23758974},
            ...     {0: 0.44941979, 1: 0.32962558, 2: 0.22095463}
            ... ]

            >>> metric = metrics.RollingCrossEntropy(window_size=2)

            >>> for y_t, y_p in zip(y_true, y_pred):
            ...     metric = metric.update(y_t, y_p)
            ...     print(metric.get())
            1.222454...
            1.169691...
            1.277069...
            1.473503...

            >>> metric
            RollingCrossEntropy: 1.473504

    """

    def update(self, y_true, y_pred):
        return super().update(optim.CrossEntropy().eval(y_true, y_pred))
