from . import base

__all__ = ["Jaccard"]


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
