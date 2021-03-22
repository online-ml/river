from river import metrics, utils

from . import base

__all__ = ["Rand"]


class Rand(base.Metric):
    """Rand Index.

    The Rand Index [^1] is a measure of the similarity between two data clusterings.
    Given a set of elements `S` and two partitions of `S` to compare, `X` and `Y`,
    define the following:

    * a, the number of paris of elements in `S` that are in the **same** subset in `X`
    and in the **same** subset in `Y`

    * b, the number of paris of elements in `S` that are in the **different** subset in `X`
    and in the **different** subset in `Y`

    * a, the number of paris of elements in `S` that are in the **same** subset in `X`
    and in the **different** subset in `Y`

    * a, the number of paris of elements in `S` that are in the **different** subset in `X`
    and in the **same** subset in `Y`

    The Rand index R, is

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
    0
    1.0
    0.3333333333333333
    0.5
    0.6
    0.6666666666666666

    References
    ----------
    [^1]: Wikipedia contributors. (2021, January 13). Rand index.
          In Wikipedia, The Free Encyclopedia,
          from https://en.wikipedia.org/w/index.php?title=Rand_index&oldid=1000098911

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

    def revert(self, y_true, y_pred, sample_weight, correction=None):

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
            return 0
