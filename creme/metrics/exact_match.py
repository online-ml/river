from . import base


__all__ = ['ExactMatch']


class ExactMatch(base.MultiOutputClassificationMetric):
    """Exact match score.

    This is the most strict multi-label metric, defined as the number of
    samples that have all their labels correctly classified, divided by the
    total number of samples.

    Examples
    --------

    >>> from creme import metrics

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

    >>> metric = metrics.ExactMatch()
    >>> for yt, yp in zip(y_true, y_pred):
    ...     metric = metric.update(yt, yp)

    >>> metric
    ExactMatch: 0.333333

    """

    @property
    def bigger_is_better(self):
        return True

    @property
    def requires_labels(self):
        return True

    def get(self):

        try:
            return self.cm.exact_match_cnt / self.cm.n_samples
        except ZeroDivisionError:
            return 0.
