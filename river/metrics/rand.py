import scipy

from river import metrics, utils

from . import base

__all__ = ["AdjustedRand", "Rand"]


class Rand(base.Metric):
    """Rand Index.

    The Rand Index [^1] [^2] is a measure of the similarity between two data clusterings.
    Given a set of elements `S` and two partitions of `S` to compare, `X` and `Y`,
    define the following:

    * a, the number of pairs of elements in `S` that are in the **same** subset in `X`
    and in the **same** subset in `Y`

    * b, the number of pairs of elements in `S` that are in the **different** subset in `X`
    and in **different** subsets in `Y`

    * c, the number of pairs of elements in `S` that are in the **same** subset in `X`
    and in **different** subsets in `Y`

    * d, the number of pairs of elements in `S` that are in the **different** subset in `X`
    and in the **same** subset in `Y`

    The Rand index, R, is

    $$
    R = \frac{a+b}{a+b+c+d} = \frac{a+b}{\frac{n(n-1)}{2}}.
    $$

    Examples
    --------
    >>> from river import metrics

    >>> y_true = [1, 1, 2, 2, 3, 3]
    >>> y_pred = [1, 1, 1, 2, 2, 2]

    >>> metric = metrics.Rand()

    >>> for yt, yp in zip(y_true, y_pred):
    ...     print(metric.update(yt, yp).get())
    0.0
    1.0
    0.3333333333333333
    0.5
    0.6
    0.6666666666666666

    >>> metric
    Rand: 0.666667

    References
    ----------
    [^1]: Wikipedia contributors. (2021, January 13). Rand index.
          In Wikipedia, The Free Encyclopedia,
          from https://en.wikipedia.org/w/index.php?title=Rand_index&oldid=1000098911
    [^2]: W. M. Rand (1971). "Objective criteria for the evaluation of clustering methods".
          Journal of the American Statistical Association. American Statistical Association.
          66 (336): 846–850. arXiv:1704.01036. doi:10.2307/2284239. JSTOR 2284239.

    """

    def __init__(self):
        super().__init__()

        self.cm = metrics.ConfusionMatrix()
        self._pairs_same_subsets = 0
        self._pairs_different_subsets = 0

    def update(self, y_true, y_pred, sample_weight=1.0):

        self.cm.update(y_true, y_pred)

        if self.cm[y_true][y_pred] >= 2:
            self._pairs_same_subsets += self.cm[y_true][y_pred] - 1

        for i in range(self.cm.shape[0]):
            for j in range(self.cm.shape[1]):
                if i != y_true and j != y_pred:
                    self._pairs_different_subsets += self.cm[i][j]

        return self

    def revert(self, y_true, y_pred, sample_weight=1.0):

        self.cm.revert(y_true, y_pred)

        if self.cm[y_true][y_pred] >= 1:
            self._pairs_same_subsets -= self.cm[y_true][y_pred]

        for i in range(self.cm.shape[0]):
            for j in range(self.cm.shape[1]):
                if i != y_true and j != y_pred:
                    self._pairs_different_subsets -= self.cm[i][j]

        return self

    @property
    def bigger_is_better(self):
        return True

    def works_with(self, model):
        return utils.inspect.isclassifier(model) or utils.inspect.isclusterer(model)

    def get(self):

        try:
            return (self._pairs_same_subsets + self._pairs_different_subsets) / (
                self.cm.n_samples * (self.cm.n_samples - 1) / 2
            )
        except ZeroDivisionError:
            return 0.0


class AdjustedRand(base.Metric):
    """Adjusted Rand Index.

    The Adjusted Rand Index is the corrected-for-chance version of the Rand Index [^1] [^2].
    Such a correction for chance establishes a baseline by using the expected similarity
    of all pair-wise comparisions between clusterings specified by a random model.

    Traditionally, the Rand Index was corrected using the Permutation Model for Clustering.
    However, the premises of the permutation model are frequently violated; in many
    clustering scenarios, either the number of clusters or the size distribution of those
    clusters vary drastically. Variations of the adjusted Rand Index account for different
    models of random clusterings.

    Though the Rand Index may only yield a value between 0 and 1, the Adjusted Rand index
    can yield negative values if the index is less than the expected index.

    Examples
    --------
    >>> from river import metrics

    >>> y_true = [1, 1, 2, 2, 3, 3]
    >>> y_pred = [1, 1, 1, 2, 2, 2]

    >>> metric = metrics.AdjustedRand()

    >>> for yt, yp in zip(y_true, y_pred):
    ...     print(metric.update(yt, yp).get())
    0.0
    0.0
    0.0
    0.0
    0.2105263157894737
    0.3333333333333333

    >>> metric
    AdjustedRand: 0.333333

    References
    ----------
    [^1]: Wikipedia contributors. (2021, January 13). Rand index.
          In Wikipedia, The Free Encyclopedia,
          from https://en.wikipedia.org/w/index.php?title=Rand_index&oldid=1000098911
    [^2]: W. M. Rand (1971). "Objective criteria for the evaluation of clustering methods".
          Journal of the American Statistical Association. American Statistical Association.
          66 (336): 846–850. arXiv:1704.01036. doi:10.2307/2284239. JSTOR 2284239.

    """

    def __init__(self):
        super().__init__()

        self.cm = metrics.ConfusionMatrix()
        self._binomial_all_entries = 0
        self._binomial_sum_rows = 0
        self._binomial_sum_cols = 0

    def update(self, y_true, y_pred, sample_weight=1.0):

        self._binomial_all_entries += scipy.special.binom(
            self.cm[y_true][y_pred] + 1, 2
        ) - scipy.special.binom(self.cm[y_true][y_pred], 2)

        self._binomial_sum_rows += scipy.special.binom(
            self.cm.sum_row[y_true] + 1, 2
        ) - scipy.special.binom(self.cm.sum_row[y_true], 2)

        self._binomial_sum_cols += scipy.special.binom(
            self.cm.sum_col[y_pred] + 1, 2
        ) - scipy.special.binom(self.cm.sum_row[y_pred], 2)

        self.cm.update(y_true, y_pred)

        return self

    def revert(self, y_true, y_pred, sample_weight=1.0):

        self._binomial_all_entries -= scipy.special.binom(
            self.cm[y_true][y_pred], 2
        ) - scipy.special.binom(self.cm[y_true][y_pred] - 1, 2)

        self._binomial_sum_rows -= scipy.special.binom(
            self.cm.sum_row[y_true], 2
        ) - scipy.special.binom(self.cm.sum_row[y_true] - 1, 2)

        self._binomial_sum_cols -= scipy.special.binom(
            self.cm.sum_col[y_pred], 2
        ) - scipy.special.binom(self.cm.sum_row[y_pred] - 1, 2)

        self.cm.revert(y_true, y_pred)

        return self

    @property
    def bigger_is_better(self):
        return True

    def works_with(self, model):
        return utils.inspect.isclassifier(model) or utils.inspect.isclusterer(model)

    def get(self):

        if self.cm.n_samples <= 2:
            return 0.0
        else:
            try:
                numerator = self._binomial_all_entries - (
                    self._binomial_sum_rows * self._binomial_sum_cols
                ) / scipy.special.binom(self.cm.n_samples, 2)
                denominator = 0.5 * (
                    self._binomial_sum_rows + self._binomial_sum_cols
                ) - (
                    self._binomial_sum_rows * self._binomial_sum_cols
                ) / scipy.special.binom(
                    self.cm.n_samples, 2
                )
                return numerator / denominator
            except ZeroDivisionError:
                return 0.0
