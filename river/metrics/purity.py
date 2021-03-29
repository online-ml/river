from river import metrics

__all__ = ["Purity"]


class Purity(metrics.MultiClassMetric):
    r"""Purity.

    In a similar fashion with Entropy, the purity of a clustering solution,
    compared to the original true label is defined to be the fraction of the
    overall cluster size that the largest class of documents assigned to that
    cluster represents. The overall purity of the clustering solution is obtained
    as a weighted sum of the individual cluster purities and is given by:

    $$
    Purity = \sum_{r=1}^k \frac{n_r}{n} \times \left( \frac{1}{n_r} \max_i (n^i_r) \right)
    = \sum_{r=1}^k \frac{1}{n} \max_i (n^i_r)
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

    >>> metric = metrics.Purity()
    >>> for yt, yp in zip(y_true, y_pred):
    ...     print(metric.update(yt, yp).get())
    1.0
    1.0
    0.6666666666666666
    0.75
    0.6
    0.6666666666666666

    >>> metric
    Purity: 0.666667

    References
    ----------
    [^1]: Ying Zhao and George Karypis. 2001. Criterion functions for
          ducument clustering: Experiments and analysis. Technical
          Report TR 01–40, Department of Computer Science, University of Minnesota.

    """

    def __init__(self, cm=None):
        super().__init__(cm)

    def get(self):

        purity = 0

        for i in self.cm.classes:
            max_entry_cluster_i = 0
            for j in self.cm.classes:
                if self.cm[j][i] > max_entry_cluster_i:
                    max_entry_cluster_i = self.cm[j][i]
            purity += max_entry_cluster_i

        return purity / self.cm.n_samples
