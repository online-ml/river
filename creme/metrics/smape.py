from . import base


class SMAPE(base.Metric):
    """Symmetric mean absolute percentage error.

    Example:

    ::

        >>> from creme import metrics

        >>> y_true = [100, 100]
        >>> y_pred = [110, 90]

        >>> metric = metrics.SMAPE()
        >>> for y_t, y_p in zip(y_true, y_pred):
        ...     print(metric.update(y_t, y_p))
        SMAPE: 4.761905
        SMAPE: 5.012531

    """

    def __init__(self):
        self.sum = 0
        self.n = 0

    def update(self, y_true, y_pred):
        if y_true != y_pred:
            self.sum += abs(y_true - y_pred) / (abs(y_true) + abs(y_pred)) / 2
        self.n += 1
        return self

    def get(self):
        if self.n:
            return 200 / self.n * self.sum
        return 0
