from . import base

__all__ = ["RegressionMultiOutput"]


class RegressionMultiOutput(base.MultiOutputRegressionMetric, base.WrapperMetric):
    """Wrapper for multi-output regression.

    This wraps a regression metric to make it compatible with multi-output regression tasks. The
    value of each output will be fed sequentially to the `get` method of the provided metric.

    Parameters
    ----------
    metric
        The regression metric to evaluate with each output.

    """

    def __init__(self, metric: "base.RegressionMetric"):
        self._metric = metric

    @property
    def metric(self):
        return self._metric

    def update(self, y_true, y_pred, sample_weight=1.0):
        for i in y_true:
            self.metric.update(y_true[i], y_pred[i], sample_weight)
        return self

    def revert(self, y_true, y_pred, sample_weight=1.0):
        for i in y_true:
            self.metric.revert(y_true[i], y_pred[i], sample_weight)
        return self

    def get(self):
        return self.metric.get()
