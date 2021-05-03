from . import base

__all__ = ["Jaccard", "SorensenDice"]


class Jaccard(base.MultiOutputClassificationMetric):
    """Jaccard index for binary multi-outputs.

    The Jaccard index, or Jaccard similarity coefficient, defined as the size of the intersection
    divided by the size of the union of two label sets, is used to compare the set of predicted
    labels for a sample with the corresponding set of labels in `y_true`.

    The Jaccard index may be a poor metric if there are no positives for some samples or labels.
    The Jaccard index is undefined if there are no true or predicted labels, this implementation
    will return a score of 0.0 if this is the case.

    Parameters
    ----------
    cm
        This parameter allows sharing the same confusion matrix between multiple metrics. Sharing a
        confusion matrix reduces the amount of storage and computation time.

    Examples
    --------

    >>> from river import metrics

    >>> y_true = [
    ...     {0: False, 1: True, 2: True},
    ...     {0: True, 1: True, 2: False},
    ... ]

    >>> y_pred = [
    ...     {0: True, 1: True, 2: True},
    ...     {0: True, 1: False, 2: False},
    ... ]

    >>> jac = metrics.Jaccard()
    >>> for yt, yp in zip(y_true, y_pred):
    ...     jac = jac.update(yt, yp)

    >>> jac
    Jaccard: 0.583333

    References
    ----------
    [^1]: [Wikipedia section on similarity of asymmetric binary attributes](https://www.wikiwand.com/en/Jaccard_index#/Similarity_of_asymmetric_binary_attributes)

    """

    @property
    def bigger_is_better(self):
        return True

    @property
    def requires_labels(self):
        return True

    def get(self):

        try:
            return self.cm.jaccard_sum / self.cm.n_samples
        except ZeroDivisionError:
            return 0.0


class SorensenDice(Jaccard):
    r"""Sørensen-Dice coefficient.

    Sørensen-Dice coefficient [^1] (or Sørensen Index, Dice's coefficient) is a statistic used to gauge
    the similarity of two samples. Sørensen's original formula was intended to be applied to discrete
    data. Given two sets, $X$ and $Y$, it is defined as:

    $$
    DSC = \frac{2 |X \cap Y|}{|X| + |Y|}.
    $$

    It is equal to twice the number of elements common to both sets divided by the sum of the number of
    elements in each set.

    The coefficient is not very different in form from the Jaccard index. The only difference between the
    two metrics is that the Jaccard index only counts true positives once in both the numerator and
    denominator. In fact, both are equivalent in the sense that given a value for the Sorensen-Dice index,
    once can canculate the respective Jaccard value and vice versa, using the equations

    $$
    \begin{equation}
    J = \frac{S}{2-S}, \\ S = \frac{2J}{1+J}.
    \end{equation}
    $$

    Parameters
    ----------
    cm
        This parameter allows sharing the same confusion matrix between multiple metrics. Sharing a
        confusion matrix reduces the amount of storage and computation time.

    Examples
    --------

    >>> from river import metrics

    >>> y_true = [
    ...     {0: False, 1: True, 2: True},
    ...     {0: True, 1: True, 2: False},
    ... ]

    >>> y_pred = [
    ...     {0: True, 1: True, 2: True},
    ...     {0: True, 1: False, 2: False},
    ... ]

    >>> sorensen_dice = metrics.SorensenDice()
    >>> for yt, yp in zip(y_true, y_pred):
    ...     sorensen_dice = sorensen_dice.update(yt, yp)

    >>> sorensen_dice
    SorensenDice: 0.736842

    References
    ----------
    [^1]: [Wikipedia article on Sørensen-Dice coefficient](https://en.wikipedia.org/wiki/Sørensen-Dice_coefficient)

    """

    def get(self):
        j = super().get()
        return 2 * j / (1 + j)
