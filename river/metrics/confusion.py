from __future__ import annotations

import functools
from collections import defaultdict

from river import utils


class ConfusionMatrix:
    """Confusion Matrix for binary and multi-class classification.

    Parameters
    ----------
    classes
        The initial set of classes. This is optional and serves only for displaying purposes.

    Examples
    --------

    >>> from river import metrics

    >>> y_true = ['cat', 'ant', 'cat', 'cat', 'ant', 'bird']
    >>> y_pred = ['ant', 'ant', 'cat', 'cat', 'ant', 'cat']

    >>> cm = metrics.ConfusionMatrix()

    >>> for yt, yp in zip(y_true, y_pred):
    ...     cm = cm.update(yt, yp)

    >>> cm
           ant  bird   cat
     ant     2     0     0
    bird     0     0     1
     cat     1     0     2

    >>> cm['bird']['cat']
    1.0

    Notes
    -----
    This confusion matrix is a 2D matrix of shape `(n_classes, n_classes)`, corresponding
    to a single-target (binary and multi-class) classification task.

    Each row represents `true` (actual) class-labels, while each column corresponds
    to the `predicted` class-labels. For example, an entry in position `[1, 2]` means
    that the true class-label is 1, and the predicted class-label is 2 (incorrect prediction).

    This structure is used to keep updated statistics about a single-output classifier's
    performance and to compute multiple evaluation metrics.

    """

    def __init__(self, classes=None):
        self._init_classes = set(classes) if classes is not None else set()
        self.sum_row = defaultdict(float)
        self.sum_col = defaultdict(float)
        self.data = defaultdict(functools.partial(defaultdict, float))
        self.n_samples = 0
        self.total_weight = 0

    def __getitem__(self, key):
        """Syntactic sugar for accessing the counts directly."""
        return self.data[key]

    def update(self, y_true, y_pred, sample_weight=1.0):
        self.n_samples += 1
        self._update(y_true, y_pred, sample_weight)
        return self

    def revert(self, y_true, y_pred, sample_weight=1.0):
        self.n_samples -= 1
        # Revert is equal to subtracting so we pass the negative sample_weight
        self._update(y_true, y_pred, -sample_weight)
        return self

    def _update(self, y_true, y_pred, sample_weight):
        self.data[y_true][y_pred] += sample_weight
        self.total_weight += sample_weight
        self.sum_row[y_true] += sample_weight
        self.sum_col[y_pred] += sample_weight

    @property
    def classes(self):
        return list(
            {c for c, n in self.sum_row.items() if n} | {c for c, n in self.sum_col.items() if n}
        )

    def __repr__(self):
        classes = sorted(self.classes)
        if not classes:
            return ""

        headers = [""] + list(map(str, classes))
        columns = [headers[1:]]
        for col in classes:
            columns.append([f"{int(self.data[row][col]):,}" for row in classes])

        return utils.pretty.print_table(headers, columns)

    def support(self, label):
        return self.sum_row[label]

    def true_positives(self, label):
        return self.data[label][label]

    def true_negatives(self, label):
        return self.total_true_positives - self.data[label][label]

    def false_positives(self, label):
        return self.sum_col[label] - self.data[label][label]

    def false_negatives(self, label):
        return self.sum_row[label] - self.data[label][label]

    @property
    def total_true_positives(self):
        return sum(self.true_positives(label) for label in self.classes)

    @property
    def total_true_negatives(self):
        return sum(self.true_negatives(label) for label in self.classes)

    @property
    def total_false_positives(self):
        return sum(self.false_positives(label) for label in self.classes)

    @property
    def total_false_negatives(self):
        return sum(self.false_negatives(label) for label in self.classes)
