from .. import utils

from . import base
from . import per_class


__all__ = ['Rolling']


class Rolling(base.WrapperMetric, utils.Window):
    """Wrapper for computing metrics over a window.

    Parameters:
        metric (metrics.Metric)

    Example:

        ::

            >>> from creme import metrics

            >>> y_true = [3, -0.5, 2, 7]
            >>> y_pred = [2.5, 0.0, 2, 8]

            >>> metric = metrics.Rolling(metrics.MSE(), window_size=2)

            >>> for y_t, y_p in zip(y_true, y_pred):
            ...     print(metric.update(y_t, y_p))
            Rolling of size 2 MSE: 0.25
            Rolling of size 2 MSE: 0.25
            Rolling of size 2 MSE: 0.125...
            Rolling of size 2 MSE: 0.5...

    """

    def __init__(self, metric, window_size):
        super().__init__(size=window_size)
        self.window_size = window_size
        self._metric = metric

    @property
    def metric(self):
        return self._metric

    def update(self, y_true, y_pred, sample_weight=1.):
        self.metric.update(y_true, y_pred, sample_weight)
        if len(self) == self.window_size:
            self.metric.revert(*self[0])
        self.append((y_true, y_pred, sample_weight))
        return self

    def revert(self, y_true, y_pred, sample_weight=1.):
        self.metric.revert(y_true, y_pred, sample_weight)
        return self

    def get(self):
        return self.metric.get()

    def __str__(self):
        if isinstance(self._metric, per_class.PerClass):
            return f'Rolling of size {self.window_size}\n' + '\n'.join((
                f'    {s}'
                for s in str(self.metric).split('\n'))
            )
        return f'Rolling of size {self.window_size} {str(self.metric)}'
