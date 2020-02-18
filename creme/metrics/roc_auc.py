from scipy import integrate

from . import base
from . import confusion


__all__ = ['ROCAUC']


class ROCAUC(base.BinaryMetric):
    """Receiving Operating Characteristic Area Under the Curve.

    Parameters:
        num_thresholds (int): The number of thresholds to use to discretize the ROC curve. The
            higher this is, the closer the output will be to the true ROC AUC value, at the cost of
            more time and memory.

    Attributes:
        thresholds (list)
        cms (list): Contains a `metrics.ConfusionMatrix` for each threshold.

    Example:

        ::

            >>> from creme import metrics

            >>> y_true = [ 0,  0,   1,  1]
            >>> y_pred = [.1, .4, .35, .8]

            >>> metric = metrics.ROCAUC()

            >>> for yt, yp in zip(y_true, y_pred):
            ...     metric = metric.update(yt, yp)

            >>> metric
            ROCAUC: 0.875

            The true ROC AUC is in fact 0.75. We can improve the accuracy by increasing the amount
            of thresholds. This comes at the cost more computation time and more memory usage.

            >>> metric = metrics.ROCAUC(num_thresholds=20)

            >>> for yt, yp in zip(y_true, y_pred):
            ...     metric = metric.update(yt, yp)

            >>> metric
            ROCAUC: 0.75

    .. warning::
        This metric is an approximation of the true ROC AUC. Computing the true ROC AUC would
        require storing all the predictions and true outputs, which isn't an option. The
        approximation error is not significant as long as the predicted probabilities are well
        calibrated.

    """

    def __init__(self, num_thresholds=10):
        self.num_thresholds = num_thresholds
        self.thresholds = [i / (num_thresholds - 1) for i in range(num_thresholds)]
        self.thresholds[0] -= 1e-7
        self.thresholds[-1] += 1e-7
        self.cms = [confusion.ConfusionMatrix() for _ in range(num_thresholds)]

    def update(self, y_true, y_pred, sample_weight=1.):
        p_true = y_pred.get(True, 0.) if isinstance(y_pred, dict) else y_pred
        for t, cm in zip(self.thresholds, self.cms):
            cm.update(y_true=bool(y_true), y_pred=p_true > t, sample_weight=sample_weight)
        return self

    def revert(self, y_true, y_pred, sample_weight=1.):
        p_true = y_pred.get(True, 0.) if isinstance(y_pred, dict) else y_pred
        for t, cm in zip(self.thresholds, self.cms):
            cm.revert(y_true=bool(y_true), y_pred=p_true > t, sample_weight=sample_weight)
        return self

    @property
    def bigger_is_better(self):
        return True

    @property
    def requires_labels(self):
        return False

    def get(self):

        tprs = [0] * self.num_thresholds
        fprs = [0] * self.num_thresholds

        def safe_div(a, b):
            try:
                return a / b
            except ZeroDivisionError:
                return 0.

        for i, cm in enumerate(self.cms):
            tp = cm.counts.get(True, {}).get(True, 0)
            tn = cm.counts.get(False, {}).get(False, 0)
            fp = cm.counts.get(False, {}).get(True, 0)
            fn = cm.counts.get(True, {}).get(False, 0)

            tprs[i] = safe_div(a=tp, b=tp + fn)
            fprs[i] = safe_div(a=fp, b=fp + tn)

        return -integrate.trapz(x=fprs, y=tprs)
