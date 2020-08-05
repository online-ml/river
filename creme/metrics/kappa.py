from . import base


__all__ = ['CohenKappa']


class CohenKappa(base.MultiClassMetric):
    """Cohen's Kappa score.

    Examples
    --------

    >>> from creme import metrics

    >>> y_true = ['cat', 'ant', 'cat', 'cat', 'ant', 'bird']
    >>> y_pred = ['ant', 'ant', 'cat', 'cat', 'ant', 'cat']

    >>> metric = metrics.CohenKappa()

    >>> for yt, yp in zip(y_true, y_pred):
    ...     metric = metric.update(yt, yp)

    >>> metric
    CohenKappa: 0.428571

    """

    def get(self):

        try:
            p0 = self.cm.sum_diag / self.cm.n_samples  # same as accuracy
        except ZeroDivisionError:
            p0 = 0

        pe = 0

        for c in self.cm.classes:
            estimation_row = self.cm.sum_row[c] / self.cm.n_samples
            estimation_col = self.cm.sum_col[c] / self.cm.n_samples
            pe += estimation_row * estimation_col

        try:
            return (p0 - pe) / (1 - pe)
        except ZeroDivisionError:
            return 0.
