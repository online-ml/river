import collections
import copy
import functools

from . import base


__all__ = ['PerClass']


class PerClass(base.WrapperMetric, base.MultiClassMetric):
    """Wrapper for computing metrics per class.

    Parameters:
        metric (metrics.BinaryMetric)

    Example:

        ::

            >>> from creme import metrics

            >>> y_true = [0, 0, 1, 1, 2, 1, 0]
            >>> y_pred = [0, 0, 0, 1, 1, 1, 1]

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
            0: Accuracy: 0.666667
            1: Accuracy: 0.666667
            2: Accuracy: 0.
            ---

        You can combine this with `metrics.Rolling` to obtain rolling metrics for each class.

        ::

            >>> from creme import metrics

            >>> y_true = [0, 0, 1, 1, 2, 1, 0]
            >>> y_pred = [0, 0, 0, 1, 1, 1, 1]

            >>> metric = metrics.Rolling(metrics.PerClass(metrics.Accuracy()), window_size=2)

            >>> for y_t, y_p in zip(y_true, y_pred):
            ...     print(metric.update(y_t, y_p))
            ...     print('---')
            Rolling of size 2
                0: Accuracy: 1.
            ---
            Rolling of size 2
                0: Accuracy: 1.
            ---
            Rolling of size 2
                0: Accuracy: 1.
                1: Accuracy: 0.
            ---
            Rolling of size 2
                0: Accuracy: 0.
                1: Accuracy: 0.5
            ---
            Rolling of size 2
                1: Accuracy: 1.
                2: Accuracy: 0.
            ---
            Rolling of size 2
                1: Accuracy: 1.
                2: Accuracy: 0.
            ---
            Rolling of size 2
                0: Accuracy: 0.
                1: Accuracy: 1.
            ---

    Note:
        Note that ``Rolling(PerClass(metric))`` will produce different results than
        ``PerClass(Rolling(metric))``.

    """

    def __init__(self, metric):
        self.metrics = collections.defaultdict(functools.partial(copy.deepcopy, metric))
        self._metric = metric
        self._class_counts = collections.Counter()

    @property
    def metric(self):
        return self._metric

    @property
    def requires_labels(self):
        return self._metric.requires_labels

    def update(self, y_true, y_pred, sample_weight=1.):
        self.metrics[y_true].update(y_true, y_pred, sample_weight)
        self._class_counts.update([y_true, y_pred])
        return self

    def revert(self, y_true, y_pred, sample_weight=1.):
        self.metrics[y_true].revert(y_true, y_pred, sample_weight)
        self._class_counts.subtract([y_true, y_pred])
        return self

    def get(self):
        return {c: m.get() for c, m in self.metrics.items() if self._class_counts[c] > 0}

    def __str__(self):
        return '\n'.join(
            f'{c}: {str(m)}'
            for c, m in self.metrics.items()
            if self._class_counts[c] > 0
        )
