from . import base


__all__ = ['Accuracy']


class Accuracy(base.MultiClassMetric):
    """Accuracy score, which is the percentage of exact matches.

    Examples
    --------

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

    def get(self):
        try:
            return self.cm.sum_diag / self.cm.n_samples
        except ZeroDivisionError:
            return 0.
