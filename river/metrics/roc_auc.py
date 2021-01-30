from scipy import integrate

from . import base, confusion

__all__ = ["ROCAUC"]


class ROCAUC(base.BinaryMetric):
    """Receiving Operating Characteristic Area Under the Curve.

    This metric is an approximation of the true ROC AUC. Computing the true ROC AUC would
    require storing all the predictions and ground truths, which isn't desirable. The approximation
    error is not significant as long as the predicted probabilities are well calibrated. In any
    case, this metric can still be used to reliably compare models between each other.

    Parameters
    ----------
    n_thresholds
        The number of thresholds used for discretizing the ROC curve. A higher value will lead to
        more accurate results, but will also cost more time and memory.
    pos_val
        Value to treat as "positive".

    Examples
    --------

    >>> from river import metrics

    >>> y_true = [ 0,  0,   1,  1]
    >>> y_pred = [.1, .4, .35, .8]

    >>> metric = metrics.ROCAUC()

    >>> for yt, yp in zip(y_true, y_pred):
    ...     metric = metric.update(yt, yp)

    >>> metric
    ROCAUC: 0.875

    The true ROC AUC is in fact 0.75. We can improve the accuracy by increasing the amount
    of thresholds. This comes at the cost more computation time and more memory usage.

    >>> metric = metrics.ROCAUC(n_thresholds=20)

    >>> for yt, yp in zip(y_true, y_pred):
    ...     metric = metric.update(yt, yp)

    >>> metric
    ROCAUC: 0.75

    """

    def __init__(self, n_thresholds=10, pos_val=True):
        self.n_thresholds = n_thresholds
        self.pos_val = pos_val
        self.thresholds = [i / (n_thresholds - 1) for i in range(n_thresholds)]
        self.thresholds[0] -= 1e-7
        self.thresholds[-1] += 1e-7
        self.cms = [confusion.ConfusionMatrix() for _ in range(n_thresholds)]

    def update(self, y_true, y_pred, sample_weight=1.0):
        p_true = y_pred.get(True, 0.0) if isinstance(y_pred, dict) else y_pred
        for t, cm in zip(self.thresholds, self.cms):
            cm.update(y_true == self.pos_val, p_true > t, sample_weight)
        return self

    def revert(self, y_true, y_pred, sample_weight=1.0):
        p_true = y_pred.get(True, 0.0) if isinstance(y_pred, dict) else y_pred
        for t, cm in zip(self.thresholds, self.cms):
            cm.revert(y_true == self.pos_val, p_true > t, sample_weight)
        return self

    @property
    def requires_labels(self):
        return False

    def get(self):

        tprs = [0] * self.n_thresholds
        fprs = [0] * self.n_thresholds

        def safe_div(a, b):
            try:
                return a / b
            except ZeroDivisionError:
                return 0.0

        for i, cm in enumerate(self.cms):
            tp = cm.true_positives(self.pos_val)
            tn = cm.true_negatives(self.pos_val)
            fp = cm.false_positives(self.pos_val)
            fn = cm.false_negatives(self.pos_val)

            tprs[i] = safe_div(a=tp, b=tp + fn)
            fprs[i] = safe_div(a=fp, b=fp + tn)

        return -integrate.trapz(x=fprs, y=tprs)
