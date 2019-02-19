from . import base


class MAE(base.Metric):
    """Mean absolute error.

    Example:

    ::

        >>> from creme import metrics

        >>> y_true = [3, -0.5, 2, 7]
        >>> y_pred = [2.5, 0.0, 2, 8]

        >>> metric = metrics.MAE()
        >>> for y_t, y_p in zip(y_true, y_pred):
        ...     print(metric.update(y_t, y_p).get())
        0.5
        0.5
        0.333333...
        0.5

    """

    def __init__(self):
        self.absolute_error = 0
        self.count = 0

    def update(self, y_true, y_pred):
        self.absolute_error += abs(y_true - y_pred)
        self.count += 1
        return self

    def get(self):
        if self.count:
            return self.absolute_error / self.count
        return 0
