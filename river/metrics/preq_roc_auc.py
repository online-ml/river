from river import metrics, utils

from .efficient_preqrocauc import EfficientPreqROCAUC

__all__ = ["PreqROCAUC"]

class PreqROCAUC(metrics.base.BinaryMetric):
    """Prequential version of the Receiving Operating Characteristic Area Under the Curve.

    The Prequential ROCAUC (predictive sequential) calculates the metric using the instances
    in its window of size S. It keeps a queue of the instances, when a instance is added
    and the queue length is equal or bigger than S, the last instance is removed. The metric
    has a tree with ordered instances, in order to calculate the metric efficiently. It was
    implemented based on the algorithm presented in Brzezinski and Stefanowski, 2017.

    Parameters
    ----------
    window_size:
        The max length of the window
    pos_val
        Value to treat as "positive"

    Examples
    --------

    >>> from river import metrics

    >>> y_true = [ 0,  1,  0,  1,  0,  1,  0,  0,   1,  1]
    >>> y_pred = [.3, .5, .5, .7, .1, .3, .1, .4, .35, .8]

    >>> metric = metrics.PreqROCAUC(window_size=4)

    >>> for yt, yp in zip(y_true, y_pred):
    ...     metric = metric.update(yt, yp)

    >>> metric
    ROCAUC: 75.00%

    """

    def __init__(self, window_size=1000, pos_val=1):
        self.window_size = window_size
        self.pos_val = pos_val
        self.metric = EfficientPreqROCAUC(pos_val, window_size)

    def works_with(self, model) -> bool:
        return (
            super().works_with(model)
            or utils.inspect.isanomalydetector(model)
            or utils.inspect.isanomalyfilter(model)
        )

    def update(self, y_true, y_pred):
        self.metric.update(y_true, y_pred)
        return self

    def revert(self, y_true, y_pred):
        self.metric.revert(y_true, y_pred)
        return self

    @property
    def requires_labels(self):
        return False

    def get(self):
        return self.metric.get()
