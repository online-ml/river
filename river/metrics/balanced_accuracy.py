from river import metrics

__all__ = ["BalancedAccuracy"]


class BalancedAccuracy(metrics.base.MultiClassMetric):
    """Balanced accuracy.

    Balanced accuracy is the average of recall obtained on each class. It is used to
    deal with imbalanced datasets in binary and multi-class classification problems.

    Parameters
    ----------
    cm
        This parameter allows sharing the same confusion
        matrix between multiple metrics. Sharing a confusion matrix reduces the amount of storage
        and computation time.

    Examples
    --------

        >>> from river import metrics
        >>> y_true = [True, False, True, True, False, True]
        >>> y_pred = [True, False, True, True, True, False]

        >>> metric = metrics.BalancedAccuracy()
        >>> for yt, yp in zip(y_true, y_pred):
        ...     metric = metric.update(yt, yp)

        >>> metric
        BalancedAccuracy: 62.50%

        >>> y_true = [0, 1, 0, 0, 1, 0]
        >>> y_pred = [0, 1, 0, 0, 0, 1]
        >>> metric = metrics.BalancedAccuracy()
        >>> for yt, yp in zip(y_true, y_pred):
        ...     metric = metric.update(yt, yp)

        >>> metric
        BalancedAccuracy: 62.50%

    """

    def get(self):
        total = 0
        for c in self.cm.classes:
            try:
                total += self.cm[c][c] / self.cm.sum_row[c]
            except ZeroDivisionError:
                continue
        try:
            score = total / len(self.cm.classes)

            return score

        except ZeroDivisionError:
            return 0.0
