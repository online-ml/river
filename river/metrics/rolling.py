import collections

from river import metrics

__all__ = ["Rolling"]


class Rolling(metrics.base.WrapperMetric):
    """Wrapper for computing metrics over a window.

    This wrapper metric allows you to apply a metric over a window of observations. Under the hood,
    a buffer with the `window_size` most recent pairs of `(y_true, y_pred)` is memorised. When the
    buffer is full, the oldest pair is removed and the `revert` method of the metric is called with
    said pair.

    You should use `metrics.Rolling` to evaluate a metric over a window of fixed sized. You can use
    `metrics.TimeRolling` to instead evaluate a metric over a period of time.

    Parameters
    ----------
    metric
        A metric.
    window_size
        The number of most recent `(y_true, y_pred)` pairs on which to evaluate the metric.

    Examples
    --------

    >>> from river import metrics

    >>> y_true = [3, -0.5, 2, 7]
    >>> y_pred = [2.5, 0.0, 2, 8]

    >>> metric = metrics.Rolling(metrics.MSE(), window_size=2)

    >>> for yt, yp in zip(y_true, y_pred):
    ...     print(metric.update(yt, yp))
    MSE: 0.25   (rolling 2)
    MSE: 0.25   (rolling 2)
    MSE: 0.125  (rolling 2)
    MSE: 0.5    (rolling 2)

    """

    def __init__(self, metric: metrics.base.Metric, window_size: int):
        self.window = collections.deque(maxlen=window_size)
        self._metric = metric

    @property
    def window_size(self):
        return self.window.maxlen

    @property
    def metric(self):
        return self._metric

    def update(self, y_true, y_pred, sample_weight=1.0):
        if len(self.window) == self.window_size:
            self.metric.revert(*self.window[0])
        self.metric.update(y_true, y_pred, sample_weight)
        self.window.append((y_true, y_pred, sample_weight))
        return self

    def revert(self, y_true, y_pred, sample_weight=1.0):
        self.metric.revert(y_true, y_pred, sample_weight)
        return self

    def __repr__(self):
        if isinstance(self.metric, metrics.ClassificationReport):
            return self.metric.__repr__()
        return f"{str(self.metric)}\t(rolling {self.window_size})"
