from . import base


__all__ = ['MSE']


class MSE(base.MeanMetric, base.RegressionMetric):
    """Mean squared error.

    Example:

        >>> from creme import metrics

        >>> y_true = [3, -0.5, 2, 7]
        >>> y_pred = [2.5, 0.0, 2, 8]

        >>> metric = metrics.MSE()

        >>> for y_t, y_p in zip(y_true, y_pred):
        ...     print(metric.update(y_t, y_p).get())
        0.25
        0.25
        0.1666
        0.375

    """

    def _eval(self, y_true, y_pred):
        return (y_true - y_pred) ** 2
