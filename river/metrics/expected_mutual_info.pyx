# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

from libc.math cimport exp, lgamma

import numpy as np
from scipy.special import gammaln

cimport cython
cimport numpy as np

np.import_array()
ctypedef np.float64_t DOUBLE


def expected_mutual_info(confusion_matrix):
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

    Note that, different form most of the implementations of other mutual information metrics,
    the expected mutual information wil be implemented using numpy arrays. This implementation
    inherits from the implementation of the expected mutual information in scikit-learn.

    Parameters
    ----------
    confusion_matrix
        This parameter allows sharing the same confusion
        matrix between multiple metrics. Sharing a confusion matrix reduces the amount of storage
        and computation time.

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

    cdef int R, C
    cdef DOUBLE N, gln_N, emi, term2, term3, gln
    cdef np.ndarray[DOUBLE] gln_a, gln_b, gln_Na, gln_Nb, gln_nij, log_Nnij
    cdef np.ndarray[DOUBLE] nijs, term1
    cdef np.ndarray[DOUBLE] log_a, log_b
    cdef np.ndarray[np.int32_t] a, b
    #cdef np.ndarray[DOUBLE, ndim=2] start, end

    N = confusion_matrix.n_samples

    a = np.array([confusion_matrix.sum_row[key] for key in confusion_matrix.classes if confusion_matrix.sum_row[key]]).astype(np.int32)
    b = np.array([confusion_matrix.sum_col[key] for key in confusion_matrix.classes if confusion_matrix.sum_col[key]]).astype(np.int32)

    # any labelling with zero entropy implies EMI = 0
    if a.size == 1 or b.size == 1:
        return 0.0

    # we do not take into consideration the order of classes in numpy arrays constructed below,
    # as they will be consistent with each other, which is enough

    cdef int val
    R = len([val for val in confusion_matrix.sum_row.values() if val != 0])
    C = len([val for val in confusion_matrix.sum_col.values() if val != 0])

    # There are three major terms to the EMI equation, which are multiplied to
    # and then summed over varying nij values.
    # Although nijs[0] will never be used, having it simplifies the indexing.
    # It will also be set to 1 to stop divide by zero warnings.

    nijs = np.arange(0, max(np.max(a), np.max(b)) + 1, dtype='float')
    nijs[0] = 1

    # calculation of the first term
    term1 = nijs / N

    # calculation of the second term
    log_a, log_b = np.log(a), np.log(b)
    log_Nnij = np.log(N) + np.log(nijs)

    # calculation of the third/final term. All elements are calculated in log space
    # to prevent overflow

    gln_a, gln_b = gammaln(a + 1), gammaln(b + 1)
    gln_Na, gln_Nb = gammaln(N - a + 1), gammaln(N - b + 1)
    gln_N = gammaln(N + 1)
    gln_nij = gammaln(nijs + 1)

    # start and end values for nij terms for each summation.
    start = np.array([[v - N + w for w in b] for v in a], dtype='int')
    start = np.maximum(start, 1)
    end = np.minimum(np.resize(a, (C, R)).T, np.resize(b, (R, C))) + 1

    # emi itself is a summation over the various values.
    expected_mutual_info = 0.0
    cdef Py_ssize_t i, j, nij
    for i in range(R):
        for j in range(C):
            for nij in range(start[i,j], end[i,j]):
                term2 = log_Nnij[nij] - log_a[i] - log_b[j]
                # Numerators are positive, denominators are negative.
                gln = (gln_a[i] + gln_b[j] + gln_Na[i] + gln_Nb[j]
                     - gln_N - gln_nij[nij] - lgamma(a[i] - nij + 1)
                     - lgamma(b[j] - nij + 1)
                     - lgamma(N - a[i] - b[j] + nij + 1))
                term3 = exp(gln)
                expected_mutual_info += (term1[nij] * term2 * term3)

    return expected_mutual_info
