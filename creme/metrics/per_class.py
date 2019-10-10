import collections
import copy
import functools

from . import base


__all__ = ['PerClass']


class PerClass(base.WrapperMetric, base.MultiClassMetric):
    """Wrapper for computing metrics per class.

    Example:

        ::

            >>> from creme import metrics

            >>> y_true = [0, 0, 1, 1, 2, 1, 0]
            >>> y_pred = [0, 0, 0, 1, 1, 1, 0]

            >>> metric = metrics.PerClass(metrics.Accuracy())

            >>> for y_t, y_p in zip(y_true, y_pred):
            ...     print(metric.update(y_t, y_p))
            ...     print('---')
            0: Accuracy: 1.
            ---
            0: Accuracy: 1.
            ---
            0: Accuracy: 1.
            1: Accuracy: 0.
            ---
            0: Accuracy: 1.
            1: Accuracy: 0.5
            ---
            0: Accuracy: 1.
            1: Accuracy: 0.5
            2: Accuracy: 0.
            ---
            0: Accuracy: 1.
            1: Accuracy: 0.666667
            2: Accuracy: 0.
            ---
            0: Accuracy: 1.
            1: Accuracy: 0.666667
            2: Accuracy: 0.
            ---

    """

    def __init__(self, metric):
        self.metrics = collections.defaultdict(functools.partial(copy.deepcopy, metric))
        self._metric = metric

    @property
    def metric(self):
        return self._metric

    @property
    def requires_labels(self):
        return self._metric.requires_labels

    def update(self, y_true, y_pred, sample_weight=1.):
        self.metrics[y_true].update(y_true, y_pred, sample_weight)
        return self

    def revert(self, y_true, y_pred, sample_weight=1.):
        self.metrics[y_true].revert(y_true, y_pred, sample_weight)
        return self

    def get(self):
        return {c: m.get() for c, m in self.metrics.items()}

    def __str__(self):
        return '\n'.join(f'{c}: {str(m)}' for c, m in self.metrics.items())
