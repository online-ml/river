import math

from . import base

__all__ = ["MatthewsCorrCoef"]


class MatthewsCorrCoef(base.MultiClassMetric):
    r"""Matthews correlation coefficient.

    The Matthews correlation coefficient (MCC) or phi coefficient is used in
    Machine Learning as a measure of the quality of classifications, introduced
    by Brian W. Matthews in 1975. The MCC is defined identically to Pearson's phi
    coefficient, introduced by Karl Pearson, also known as the Yule phi coefficient
    from its introduction by Udny Yule in 1912.

    The coefficient takes into account true and false positives and negatives and is
    generally regarded as a balanced measure which can be used even if the classes
    are of very different sizes. It returns a value between -1 and 1, with 1 being
    a perfect prediction, 0 no better than random prediction and -1 means a total
    disagreement between prediction and observation.

    The MCC can be calculated directly from the (pair) confusion matrix using the original
    formula by Matthews. Let

    $$
    \begin{cases}
    N = TN + TP + FN + FP = \frac{n(n-1)}{2}, \\
    S = \frac{TP + FN}{N}, \\
    P = \frac{TP + FP}{N}.
    \end{cases}
    $$

    The MCC would be then equal to

    $$
    MCC = \frac{TP / N - S \times P}{\sqrt{PS(1 - S)(1 - P)}}.
    $$

    Parameters
    ----------
    cm
        This parameter allows sharing the same confusion
        matrix between multiple metrics. Sharing a confusion matrix reduces the amount of storage
        and computation time.

    Examples
    --------
    >>> from river import metrics

    >>> y_true = [1, 1, 2, 2, 3, 3]
    >>> y_pred = [1, 1, 1, 2, 2, 2]

    >>> metric = metrics.MatthewsCorrCoef()

    >>> for yt, yp in zip(y_true, y_pred):
    ...     print(metric.update(yt, yp).get())
    0.0
    0.0
    0.0
    0.5773502691896258
    0.36084391824351614
    0.28867513459481287

    >>> metric
    MatthewsCorrCoef: 0.288675

    References
    ----------
    [^1]: Wikipedia contributors. (2021, March 26). Matthews correlation coefficient.
          In Wikipedia, The Free Encyclopedia,
          from https://en.wikipedia.org/w/index.php?title=Matthews_correlation_coefficient&oldid=1014254295
    [^2]: Jurman, G., Riccadonna, S., & Furlanello, C. (2012).
          A Comparison of MCC and CEN Error Measures in Multi-Class Prediction.
          Plos ONE, 7(8), e41882. doi: 10.1371/journal.pone.0041882

    """

    def __init__(self, cm=None):
        super().__init__(cm)

    def get(self):

        n_correct = self.cm.sum_diag

        n_samples = sum(self.cm.sum_row.values())

        cov_ytrue_ypred = n_correct * n_samples - sum(
            self.cm.sum_col[i] * self.cm.sum_row[i] for i in self.cm.classes
        )

        cov_ypred_ypred = n_samples * n_samples - sum(
            self.cm.sum_col[i] * self.cm.sum_col[i] for i in self.cm.classes
        )

        cov_ytrue_ytrue = n_samples * n_samples - sum(
            self.cm.sum_row[i] * self.cm.sum_row[i] for i in self.cm.classes
        )

        try:
            return cov_ytrue_ypred / math.sqrt(cov_ytrue_ytrue * cov_ypred_ypred)
        except (ZeroDivisionError, ValueError):
            return 0.0
