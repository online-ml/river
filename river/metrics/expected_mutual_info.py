from __future__ import annotations

from river import metrics
from river.stats import _rust_stats


def expected_mutual_info(confusion_matrix: metrics.ConfusionMatrix) -> float:
    r"""Expected Mutual Information.

    Like the Rand index, the baseline value of mutual information between two
    random clusterings is not necessarily a constant value, and tends to become
    larger when the two partitions have a higher number of cluster (with a fixed
    number of samples $N$). Using a hypergeometric model of randomness, it can be
    shown that the expected mutual information between two random clusterings [^1] is:

    $$
    E(MI(U, V)) = \sum_{i=1}^R \sum_{i=1}^C \sum{n_{ij}=(a_i+b_j-N)^+}^{\min(a_i,b_j)} \frac{n_{ij}}{N} \log(\frac{N n_{ij}}{a_i b_j}) \frac{a_i! b_j! (N-a_i)! (N-b_j)!}{N! n_{ij}! (a_i - n_{ij})! (b_j - n_{ij})! (N - a_i - b_j + n_{ij})!}}
    $$

    where

    * $(a_i + b_j - N)^+$ denotes $\max(1, a_i + b_j - N)$,

    * $a_i$ is the sum of row i-th of the contingency table, and

    * $b_j$ is the sum of column j-th of the contingency table.

    The Adjusted Mutual Information score (AMI) relies on the Expected Mutual Information (EMI) score.

    From the formula, we note that this metric is very expensive to calculate. As such,
    the AMI will be one order of magnitude slower than most other implemented metrics.

    Parameters
    ----------
    confusion_matrix
        This parameter allows sharing the same confusion matrix between multiple metrics.
        Sharing a confusion matrix reduces the amount of storage and computation time.

    References
    ----------
    [^1]: Wikipedia contributors. (2021, March 17). Mutual information.
          In Wikipedia, The Free Encyclopedia,
          from https://en.wikipedia.org/w/index.php?title=Mutual_information&oldid=1012714929
    [^2]: Andrew Rosenberg and Julia Hirschberg (2007).
          V-Measure: A conditional entropy-based external cluster evaluation measure.
          Proceedings of the 2007 Joing Conference on Empirical Methods in Natural Language
          Processing and Computational Natural Language Learning, pp. 410 - 420,
          Prague, June 2007.
    """
    a = [int(v) for v in confusion_matrix.sum_row.values() if v]
    b = [int(v) for v in confusion_matrix.sum_col.values() if v]
    return _rust_stats.expected_mutual_info(confusion_matrix.n_samples, a, b)
