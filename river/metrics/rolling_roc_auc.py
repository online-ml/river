from __future__ import annotations

from river import metrics
from river._river_rust.stats import RsRollingROCAUC
from river.anomaly.base import AnomalyDetector, AnomalyFilter

__all__ = ["RollingROCAUC"]


class RollingROCAUC(metrics.base.BinaryMetric):
    """Rolling version of the Receiver Operating Characteristic Area Under the Curve.

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
    ...     metric.update(yt, yp)

    >>> metric
    RollingROCAUC: 75.00%

    """

    def __init__(self, window_size=1000, pos_val=True):
        self.window_size = window_size
        self.pos_val = pos_val
        self._metric = RsRollingROCAUC(int(pos_val), window_size)
        # Crossing the Rust FFI boundary costs roughly 1 µs per call, so individual
        # `update` calls would dominate the runtime. Buffering them in Python and
        # flushing in batch via `update_many` cuts the per-event cost by ~10x.
        self._buf_labels: list[int] = []
        self._buf_scores: list[float] = []

    def works_with(self, model) -> bool:
        return (
            super().works_with(model)
            or isinstance(model, AnomalyDetector)
            or isinstance(model, AnomalyFilter)
        )

    def _flush(self):
        if self._buf_labels:
            self._metric.update_many(self._buf_labels, self._buf_scores)
            self._buf_labels = []
            self._buf_scores = []

    def update(self, y_true, y_pred):
        p_true = y_pred.get(True, 0.0) if isinstance(y_pred, dict) else y_pred
        self._buf_labels.append(int(y_true))
        self._buf_scores.append(p_true)

    def revert(self, y_true, y_pred):
        # Revert needs to operate against the materialized window — the entry it
        # targets may already be in Rust state, so the buffer is flushed first.
        self._flush()
        p_true = y_pred.get(True, 0.0) if isinstance(y_pred, dict) else y_pred
        self._metric.revert(int(y_true), p_true)

    @property
    def requires_labels(self) -> bool:
        return False

    @property
    def works_with_weights(self) -> bool:
        return False

    def get(self):
        self._flush()
        return self._metric.get()
