import collections

from river import metrics


class PairConfusionMatrix(metrics.ClassificationMetric):
    r"""Pair Confusion Matrix.

    The pair confusion matrix $C$ is a 2 by 2 similarity matrix between two
    clusterings by considering all pairs of samples and counting pairs that are
    assigned into the same or into different clusters under the true and predicted
    clusterings.

    The pair confusion matrix has the following entries:

    * $C[0][0]$ (**True Negatives**): number of pairs of points that are in different clusters
    in both true and predicted labels

    * $C[0][1]$ (**False Positives**): number of pairs of points that are in the same cluster
    in predicted labels but not in predicted labels;

    * $C[1][0]$ (**False Negatives**): number of pairs of points that are in the same cluster
    in true labels but not in predicted labels;

    * $C[1][1]$ (**True Positives**): number of pairs of points that are in the same cluster
    in both true and predicted labels.

    We can also show that the four counts have the following property

    $$
    TP + FP + FN + TV = \frac{n(n-1)}{2}
    $$

    Parameters
    ----------
    cm
        This parameter allows sharing the same confusion
        matrix between multiple metrics. Sharing a confusion matrix reduces the amount of storage
        and computation time.

    Parameters
    ----------
    cm
        This parameter allows sharing the same confusion
        matrix between multiple metrics. Sharing a confusion matrix reduces the amount of storage
        and computation time.

    Examples
    --------

    >>> from river import metrics

    >>> y_true = [0, 1, 2, 2, 2]
    >>> y_pred = [0, 0, 2, 2, 1]

    >>> matrix = metrics.PairConfusionMatrix()

    >>> for yt, yp in zip(y_true, y_pred):
    ...     matrix.update(yt, yp)

    >>> matrix
    PairConfusionMatrix: {0: defaultdict(<class 'int'>, {0: 6.0, 1: 1.0}), 1: defaultdict(<class 'int'>, {0: 2.0, 1: 1.0})}

    """

    def __init__(self, cm=None):
        super().__init__(cm)

    def get(self):

        pair_confusion_matrix = {i: collections.defaultdict(int) for i in range(2)}

        sum_squares = 0
        for i in self.cm.classes:
            for j in self.cm.classes:
                sum_squares += self.cm[i][j] * self.cm[i][j]

        true_positives = (sum_squares - self.cm.n_samples) / 2

        false_positives = 0
        for i in self.cm.classes:
            for j in self.cm.classes:
                try:
                    false_positives += self.cm[i][j] * self.cm.sum_col[j]
                except KeyError:
                    continue
        false_positives = (false_positives - sum_squares) / 2

        false_negatives = 0
        for i in self.cm.classes:
            for j in self.cm.classes:
                try:
                    false_negatives += self.cm[j][i] * self.cm.sum_row[j]
                except KeyError:
                    continue
        false_negatives = (false_negatives - sum_squares) / 2

        true_negatives = (
            self.cm.n_samples * self.cm.n_samples
            - 2 * (false_positives + false_negatives)
            - sum_squares
        ) / 2

        pair_confusion_matrix[0][0] = true_negatives
        pair_confusion_matrix[0][1] = false_positives
        pair_confusion_matrix[1][0] = false_negatives
        pair_confusion_matrix[1][1] = true_positives

        return pair_confusion_matrix

    @property
    def bigger_is_better(self):
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}: {self.get()}"
