from __future__ import annotations

from river import metrics, utils

from .efficient_rollingrocauc import EfficientRollingROCAUC

__all__ = ["RollingROCAUC"]


class RollingROCAUC(metrics.base.BinaryMetric):
    """Rolling version of the Receiving Operating Characteristic Area Under the Curve.

    The RollingROCAUC calculates the metric using the instances in its window
    of size S. It keeps a queue of the instances, when an instance is added and
    the queue length is equal to S, the last instance is removed. The metric has
    a tree with ordered instances, in order to calculate the AUC efficiently.
    It was implemented based on the algorithm presented in Brzezinski and
    Stefanowski, 2017.

    The difference between this metric and the standard ROCAUC is that the latter
    calculates an approximation of the real metric considering all data from the
    beginning of the stream, while the RollingROCAUC calculates the exact value
    considering only the last S instances. This approach may be beneficial if
    it's necessary to evaluate the model's performance over time, since
    calculating the metric using the entire stream may hide the current
    performance of the classifier.

    Parameters
    ----------
    window_size
        The max length of the window.
    pos_val
        Value to treat as "positive".

    Examples
    --------

    >>> from river import metrics

    >>> y_true = [ 0,  1,  0,  1,  0,  1,  0,  0,   1,  1]
    >>> y_pred = [.3, .5, .5, .7, .1, .3, .1, .4, .35, .8]

    >>> metric = metrics.RollingROCAUC(window_size=4)

    >>> for yt, yp in zip(y_true, y_pred):
    ...     metric = metric.update(yt, yp)

    >>> metric
    RollingROCAUC: 75.00%

    """

    def __init__(self, window_size=1000, pos_val=True):
        self.window_size = window_size
        self.pos_val = pos_val
        self.__metric = EfficientRollingROCAUC(pos_val, window_size)

    def works_with(self, model) -> bool:
        return (
            super().works_with(model)
            or utils.inspect.isanomalydetector(model)
            or utils.inspect.isanomalyfilter(model)
        )

    def update(self, y_true, y_pred):
        p_true = y_pred.get(True, 0.0) if isinstance(y_pred, dict) else y_pred
        self.__metric.update(y_true, p_true)
        return self

    def revert(self, y_true, y_pred):
        p_true = y_pred.get(True, 0.0) if isinstance(y_pred, dict) else y_pred
        self.__metric.revert(y_true, p_true)
        return self

    @property
    def requires_labels(self) -> bool:
        return False

    @property
    def works_with_weights(self) -> bool:
        return False

    def get(self):
        return self.__metric.get()
