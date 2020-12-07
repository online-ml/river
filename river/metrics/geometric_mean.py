from . import base

import numpy as np

from scipy import stats


__all__ = ["GeometricMean"]


class GeometricMean(base.MultiClassMetric):
    r"""Geometric mean score.

    The geometric mean is a good indicator of a classifier's performance in the presence of class
    imbalance because it is independent of the distribution of examples between classes. This
    implementation computes the geometric mean of class-wise sensitivity (recall).

    $$
    gm = \sqrt[n]{s_1\cdot s_2\cdot s_3\cdot \ldots\cdot s_n}
    $$

    where $s_i$ is the sensitivity (recall) of class $i$ and $n$ is the
    number of classes.

    Parameters
    ----------
    cm
        This parameter allows sharing the same confusion matrix between multiple metrics. Sharing a
        confusion matrix reduces the amount of storage and computation time.

    Examples
    --------

    >>> from river import metrics

    >>> y_true = ['cat', 'ant', 'cat', 'cat', 'ant', 'bird', 'bird']
    >>> y_pred = ['ant', 'ant', 'cat', 'cat', 'ant', 'cat', 'bird']

    >>> metric = metrics.GeometricMean()

    >>> for yt, yp in zip(y_true, y_pred):
    ...     metric = metric.update(yt, yp)

    >>> metric
    GeometricMean: 0.693361

    References
    ----------
    [^1]: Barandela, R. et al. “Strategies for learning in class imbalance problems”, Pattern Recognition, 36(3), (2003), pp 849-851.

    """

    def get(self):

        if self.cm.n_classes > 0:
            sensitivity_per_class = np.zeros(self.cm.n_classes, np.float)
            for i, c in enumerate(self.cm.classes):
                try:
                    sensitivity_per_class[i] = self.cm[c][c] / self.cm.sum_row[c]
                except ZeroDivisionError:
                    continue
            with np.errstate(divide="ignore", invalid="ignore"):
                return stats.gmean(sensitivity_per_class)
        return 0.0
