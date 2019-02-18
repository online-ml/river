from . import base


class Accuracy(base.Metric):
    """Accuracy score, which is the percentage of exact matches.

    Example:

    ::

        >>> from creme import metrics

        >>> y_true = [0, 1, 2, 3]
        >>> y_pred = [0, 2, 1, 3]

        >>> metric = metrics.Accuracy()
        >>> for y_t, y_p in zip(y_true, y_pred):
        ...     print(metric.update(y_t, y_p).get())
        1.0
        0.5
        0.333333...
        0.5

    """

    def __init__(self):
        self.positives = 0
        self.count = 0

    def update(self, y_true, y_pred):
        self.positives += y_true == y_pred
        self.count += 1
        return self

    def get(self):
        if self.count:
            return self.positives / self.count
        return 0
