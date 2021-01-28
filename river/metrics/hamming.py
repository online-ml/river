import numpy as np

from . import base

__all__ = ["Hamming"]


class Hamming(base.MultiOutputClassificationMetric):
    """Hamming score.

    The Hamming score is the fraction of labels that are correctly predicted.

    Parameters
    ----------
    cm
        This parameter allows sharing the same confusion matrix between multiple metrics. Sharing a
        confusion matrix reduces the amount of storage and computation time.

    Examples
    --------

    >>> from river import metrics

    >>> y_true = [
    ...     {0: False, 1: True, 2: True},
    ...     {0: True, 1: True, 2: False},
    ...     {0: True, 1: True, 2: False},
    ... ]

    >>> y_pred = [
    ...     {0: True, 1: True, 2: True},
    ...     {0: True, 1: False, 2: False},
    ...     {0: True, 1: True, 2: False},
    ... ]

    >>> metric = metrics.Hamming()
    >>> for yt, yp in zip(y_true, y_pred):
    ...     metric = metric.update(yt, yp)

    >>> metric
    Hamming: 0.555556

    """

    @property
    def bigger_is_better(self):
        return True

    @property
    def requires_labels(self):
        return True

    def get(self):

        try:
            return np.sum(self.cm.data[:, 1, 1]) / (
                self.cm.n_samples * self.cm.n_labels
            )
        except ZeroDivisionError:
            return 0.0


class HammingLoss(base.MultiOutputClassificationMetric):
    """Hamming loss score.

    The Hamming loss is the complement of the Hamming score.

    Parameters
    ----------
    cm
        This parameter allows sharing the same confusion matrix between multiple metrics. Sharing a
        confusion matrix reduces the amount of storage and computation time.

    Examples
    --------

    >>> from river import metrics

    >>> y_true = [
    ...     {0: False, 1: True, 2: True},
    ...     {0: True, 1: True, 2: False},
    ...     {0: True, 1: True, 2: False},
    ... ]

    >>> y_pred = [
    ...     {0: True, 1: True, 2: True},
    ...     {0: True, 1: False, 2: False},
    ...     {0: True, 1: True, 2: False},
    ... ]

    >>> metric = metrics.HammingLoss()
    >>> for yt, yp in zip(y_true, y_pred):
    ...     metric = metric.update(yt, yp)

    >>> metric
    HammingLoss: 0.444444

    """

    @property
    def bigger_is_better(self):
        return True

    @property
    def requires_labels(self):
        return True

    def get(self):

        try:
            return 1.0 - np.sum(self.cm.data[:, 1, 1]) / (
                self.cm.n_samples * self.cm.n_labels
            )
        except ZeroDivisionError:
            return 0.0
