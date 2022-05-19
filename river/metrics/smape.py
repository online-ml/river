from river import metrics

__all__ = ["SMAPE"]


class SMAPE(metrics.base.MeanMetric, metrics.base.RegressionMetric):
    """Symmetric mean absolute percentage error.

    Examples
    --------

    >>> from river import metrics

    >>> y_true = [0, 0.07533, 0.07533, 0.07533, 0.07533, 0.07533, 0.07533, 0.0672, 0.0672]
    >>> y_pred = [0, 0.102, 0.107, 0.047, 0.1, 0.032, 0.047, 0.108, 0.089]

    >>> metric = metrics.SMAPE()
    >>> for yt, yp in zip(y_true, y_pred):
    ...     metric = metric.update(yt, yp)

    >>> metric
    SMAPE: 37.869392

    """

    def _eval(self, y_true, y_pred):
        den = abs(y_true) + abs(y_pred)
        if den == 0:
            return 0.0
        return 2.0 * abs(y_true - y_pred) / den

    def get(self):
        return 100 * super().get()
