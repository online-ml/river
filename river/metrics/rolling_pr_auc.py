from __future__ import annotations

from river import metrics, utils

from .efficient_rollingprauc import EfficientRollingPRAUC

__all__ = ["RollingPRAUC"]


class RollingPRAUC(metrics.base.BinaryMetric):
    """Rolling version of the Area Under the Precision-Recall Area Under Curve
    metric.

    The RollingPRAUC calculates the AUC-PR using the instances in its window
    of size S. It keeps a queue of the instances, when an instance is added
    and the queue length is equal to S, the last instance is removed.

    The AUC-PR is suitable for evaluating models under unbalanced environments.
    For now, the implementation can deal only with binary scenarios.

    Internally, this class maintains a self-balancing binary search tree to
    efficiently and precisely (i.e., the result is not an approximation)
    compute the AUC-PR considering the current window.

    This implementation is based on the paper "Efficient Prequential AUC-PR
    Computation" (Gomes, Grégio, Alves, and Almeida, 2023):
    https://doi.org/10.1109/ICMLA58977.2023.00335.


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

    >>> metric = metrics.RollingPRAUC(window_size=4)

    >>> for yt, yp in zip(y_true, y_pred):
    ...     metric.update(yt, yp)

    >>> metric
    RollingPRAUC: 83.33%

    """

    def __init__(self, window_size=1000, pos_val=True):
        self.window_size = window_size
        self.pos_val = pos_val
        self._metric = EfficientRollingPRAUC(pos_val, window_size)

    def works_with(self, model) -> bool:
        return (
            super().works_with(model)
            or utils.inspect.isanomalydetector(model)
            or utils.inspect.isanomalyfilter(model)
        )

    def update(self, y_true, y_pred):
        p_true = y_pred.get(True, 0.0) if isinstance(y_pred, dict) else y_pred
        self._metric.update(y_true, p_true)

    def revert(self, y_true, y_pred):
        p_true = y_pred.get(True, 0.0) if isinstance(y_pred, dict) else y_pred
        self._metric.revert(y_true, p_true)

    @property
    def requires_labels(self) -> bool:
        return False

    @property
    def works_with_weights(self) -> bool:
        return False

    def get(self):
        return self._metric.get()
