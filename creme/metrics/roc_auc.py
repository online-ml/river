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
        cms (list): Contains the `metrics.ConfusionMatrix` for each threshold.

    .. warning::
        This metric is an approximation of the true ROC AUC. Computing the true ROC AUC would
        require storing all the predictions and true outputs, which isn't an option. The
        approximation error is not significant as long as the predicted probabilities are well
        calibrated.

    """

    def __init__(self, num_thresholds=10):
        self.num_thresholds = num_thresholds
        self.thresholds = [i / (num_thresholds - 1) for i in range(num_thresholds)]
        self.cms = [confusion.ConfusionMatrix() for _ in range(num_thresholds)]

    def update(self, y_true, y_pred, sample_weight=1.):
        for t, cm in zip(self.thresholds, self.cms):
            cm.update(y_true=y_true, y_pred=y_pred.get(True, 0.) > t, sample_weight=1.)
        return self

    def revert(self, y_true, y_pred):
        for t, cm in zip(self.thresholds, self.cms):
            cm.update(y_true=y_true, y_pred=y_pred.get(True, 0.) > t, sample_weight=1.)
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

        for i, cm in enumerate(reversed(self.cms)):
            tp = cm.counts.get(True, {}).get(True, 0)
            tn = cm.counts.get(False, {}).get(False, 0)
            fp = cm.counts.get(False, {}).get(True, 0)
            fn = cm.counts.get(True, {}).get(False, 0)

            tprs[i] = safe_div(tp, tp + fn)
            fprs[i] = safe_div(fp, fp + tn)

        # Remove duplicate values
        tprs, fprs = zip(*[(tprs[0], fprs[0])] + [
            (tprs[i + 1], b)
            for i, (a, b) in enumerate(zip(fprs[:-1], fprs[1:]))
            if a != b
        ])

        return integrate.simps(x=fprs, y=tprs)
