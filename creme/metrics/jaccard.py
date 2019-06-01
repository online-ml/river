from .. import stats

from . import base


__all__ = ['Jaccard']


class Jaccard(stats.Mean, base.MultiOutputClassificationMetric):
    """Jaccard index for binary multi-outputs.

    Example:

        ::

            >>> from creme import metrics

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

    References:
        1. `Similarity of asymmetric binary attributes <https://www.wikiwand.com/en/Jaccard_index#/Similarity_of_asymmetric_binary_attributes>`_

    """

    def bigger_is_better(self):
        return True

    def requires_labels(self):
        return True

    def update(self, y_true, y_pred):

        one_and_ones = 0
        zero_or_ones = 0

        for i in y_true:
            if y_true[i] and y_pred[i]:
                one_and_ones += 1
            elif y_true[i] or y_pred[i]:
                zero_or_ones += 1

        super().update(one_and_ones / (zero_or_ones + one_and_ones))

        return self
