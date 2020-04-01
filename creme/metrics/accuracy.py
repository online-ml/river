from . import base


__all__ = ['Accuracy']


class Accuracy(base.MeanMetric, base.MultiClassMetric):
    """Accuracy score, which is the percentage of exact matches.

    Example:

        ::

            >>> import math
            >>> from creme import metrics

            >>> y_true = [True, False, True, True, True]
            >>> y_pred = [True, True, False, True, True]

            >>> metric = metrics.Accuracy()
            >>> for i, (y_t, y_p) in enumerate(zip(y_true, y_pred)):
            ...     metric = metric.update(y_t, y_p)

            >>> metric
            Accuracy: 60.00%

    """

    fmt = '.2%'  # Will output a percentage, e.g. 0.427 will become "42,7%"

    @property
    def bigger_is_better(self):
        return True

    @property
    def requires_labels(self):
        return True

    def _eval(self, y_true, y_pred):
        return y_true == y_pred
