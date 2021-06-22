import math

from . import base

__all__ = ["VariationInfo"]


class VariationInfo(base.MultiClassMetric):
    r"""Variation of Information.

    Variation of Information (VI) [^1] [^2] is an information-based clustering measure.
    It is presented as a distance measure for comparing partitions (or clusterings)
    of the same data. It therefore does not distinguish between hypothesised and
    target clustering. VI has a number of useful properties, as follows

    * VI satisifes the metric axioms

    * VI is convexly additive. This means that, if a cluster is split, the distance
    from the new cluster to the original is the distance induced by the split times
    the size of the cluster. This guarantees that all changes to the metrics are "local".

    * VI is not affected by the number of data points in the cluster. However, it is bounded
    by the logarithm of the maximum number of clusters in true and predicted labels.

    The Variation of Information is calculated using the following formula

    $$
    VI(C, K) = H(C) + H(K) - 2 H(C, K) = H(C|K) + H(K|C)
    $$

    The bound of the variation of information [^3] can be written in terms of the number of elements,
    $VI(C, K) \leq \log(n)$, or with respect to the maximum number of clusters $K^*$,
    $VI(C, K) \leq 2 \log(K^*)$.

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

    >>> metric = metrics.VariationInfo()

    >>> for yt, yp in zip(y_true, y_pred):
    ...     print(metric.update(yt, yp).get())
    0.0
    0.0
    0.9182958340544896
    1.1887218755408673
    1.3509775004326938
    1.2516291673878228

    >>> metric
    VariationInfo: 1.251629

    References
    ----------
    [^1]: Andrew Rosenberg and Julia Hirschberg (2007).
          V-Measure: A conditional entropy-based external cluster evaluation measure.
          Proceedings of the 2007 Joing Conference on Empirical Methods in Natural Language
          Processing and Computational Natural Language Learning, pp. 410 - 420,
          Prague, June 2007.
    [^2]: Marina Meila and David Heckerman. 2001.
          An experimental comparison of model-based clustering methods.
          Mach. Learn., 42(1/2):9–29.
    [^3]: Wikipedia contributors. (2021, February 18).
          Variation of information. In Wikipedia, The Free Encyclopedia,
          from https://en.wikipedia.org/w/index.php?title=Variation_of_information&oldid=1007562715

    """

    def __init__(self, cm=None):
        super().__init__(cm)

    def get(self):

        conditional_entropy_c_k = 0.0

        conditional_entropy_k_c = 0.0

        for i in self.cm.classes:

            for j in self.cm.classes:

                try:
                    conditional_entropy_c_k -= (
                        self.cm[j][i]
                        / self.cm.n_samples
                        * math.log(self.cm[j][i] / self.cm.sum_col[i], 2)
                    )
                except (ValueError, ZeroDivisionError):
                    pass

                try:
                    conditional_entropy_k_c -= (
                        self.cm[i][j]
                        / self.cm.n_samples
                        * math.log(self.cm[i][j] / self.cm.sum_row[i], 2)
                    )
                except (ValueError, ZeroDivisionError):
                    pass

        return conditional_entropy_c_k + conditional_entropy_k_c
