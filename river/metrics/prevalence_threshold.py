import math

from river import metrics

__all__ = ["PrevalenceThreshold"]


class PrevalenceThreshold(metrics.BinaryMetric):
    r"""Prevalence Threshold (PT).

    The relationship between a positive predicted value and its target prevalence
    is propotional - though not linear in all but a special case. In consequence,
    there is a point of local extrema and maximum curvature defined only as a function
    of the sensitivity and specificity beyond which the rate of change of a test's positive
    predictive value drops at a differential pace relative to the disease prevalence.
    Using differential equations, this point was first defined by Balayla et al. [^1] and
    is termed the **prevalence threshold** (\phi_e).

    The equation for the prevalence threshold [^2] is given by the following formula

    $$
    \phi_e = \frac{\sqrt{TPR(1 - TNR)} + TNR - 1}{TPR + TNR - 1}
    $$

    with

    $$
    TPR = \frac{TP}{P} = \frac{TP}{TP + FN}, TNR = = \frac{TN}{N} = \frac{TN}{TN + FP}
    $$

    Parameters
    ----------
    cm
        This parameter allows sharing the same confusion
        matrix between multiple metrics. Sharing a confusion matrix reduces the amount of storage
        and computation time.
    pos_val
        Value to treat as "positive".

    Examples
    --------

    >>> from river import metrics

    >>> y_true = [False, False, False, True, True, True]
    >>> y_pred = [False, False, True, True, False, True]

    >>> metric = metrics.PrevalenceThreshold()
    >>> for yt, yp in zip(y_true, y_pred):
    ...     print(metric.update(yt, yp).get())
    0.0
    0.0
    1.0
    0.36602540378443876
    0.44948974278317827
    0.41421356237309503

    >>> metric
    PrevalenceThreshold: 0.414214

    References
    ----------
    [^1]: Balayla, J. (2020). Prevalence threshold ($\phi$_e) and the geometry of screening curves.
          PLOS ONE, 15(10), e0240215. DOI: 10.1371/journal.pone.0240215
    [^2]: Wikipedia contributors. (2021, March 19). Sensitivity and specificity.
          In Wikipedia, The Free Encyclopedia,
          from https://en.wikipedia.org/w/index.php?title=Sensitivity_and_specificity&oldid=1013004476

    """

    def __init__(self, cm=None, pos_val=True):
        super().__init__(cm, pos_val)

    def get(self):

        try:
            tpr = self.cm.true_positives(self.pos_val) / (
                self.cm.true_positives(self.pos_val)
                + self.cm.false_negatives(self.pos_val)
            )
        except ZeroDivisionError:
            tpr = 0.0

        try:
            tnr = self.cm.true_negatives(self.pos_val) / (
                self.cm.true_negatives(self.pos_val)
                + self.cm.false_positives(self.pos_val)
            )
        except ZeroDivisionError:
            tnr = 0.0

        try:
            return (math.sqrt(tpr * (1 - tnr)) + tnr - 1) / (tpr + tnr - 1)
        except (ZeroDivisionError, ValueError):
            return 0.0
