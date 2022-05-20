from river import metrics

__all__ = ["MAE"]


class MAE(metrics.base.MeanMetric, metrics.base.RegressionMetric):
    """Mean absolute error.

    Examples
    --------

    >>> from river import metrics

    >>> y_true = [3, -0.5, 2, 7]
    >>> y_pred = [2.5, 0.0, 2, 8]

    >>> metric = metrics.MAE()

    >>> for yt, yp in zip(y_true, y_pred):
    ...     print(metric.update(yt, yp).get())
    0.5
    0.5
    0.333
    0.5

    >>> metric
    MAE: 0.5

    """

    def _eval(self, y_true, y_pred):
        return abs(y_true - y_pred)
