from . import base


__all__ = ['Accuracy']


class Accuracy(base.MeanMetric, base.MultiClassMetric):
    """Accuracy score, which is the percentage of exact matches.

    Example:

        >>> from creme import metrics

        >>> y_true = [True, False, True, True, True]
        >>> y_pred = [True, True, False, True, True]

        >>> metric = metrics.Accuracy()
        >>> for yt, yp in zip(y_true, y_pred):
        ...     metric = metric.update(yt, yp)

        >>> metric
        Accuracy: 60.00%

    """

    _fmt = '.2%'  # will output a percentage, e.g. 0.427 will become "42,7%"

    @property
    def bigger_is_better(self):
        return True

    @property
    def requires_labels(self):
        return True

    def _eval(self, y_true, y_pred):
        return y_true == y_pred
