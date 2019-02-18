from . import base


class MSE(base.Metric):
    """Exact mean squared error.

    Example:

    ::

        >>> from creme import metrics

        >>> y_true = [3, -0.5, 2, 7]
        >>> y_pred = [2.5, 0.0, 2, 8]

        >>> mse = metrics.MSE()
        >>> for y_t, y_p in zip(y_true, y_pred):
        ...     print(mse.update(y_t, y_p).get())
        0.25
        0.25
        0.1666666...
        0.375

    """

    def __init__(self):
        self.squared_error = 0
        self.count = 0

    def update(self, y_true, y_pred):
        self.squared_error += (y_true - y_pred) ** 2
        self.count += 1
        return self

    def get(self):
        if self.count:
            return self.squared_error / self.count
        return 0
