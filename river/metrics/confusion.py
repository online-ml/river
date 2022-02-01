import functools
import textwrap
from collections import defaultdict


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
        self.sum_diag = 0.0
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

        if y_true == y_pred:
            self.sum_diag += sample_weight
        self.sum_row[y_true] += sample_weight
        self.sum_col[y_pred] += sample_weight

    @property
    def classes(self):
        return list(
            set(c for c, n in self.sum_row.items() if n)
            | set(c for c, n in self.sum_col.items() if n)
        )

    @property
    def n_classes(self):
        return len(self.classes)

    def __repr__(self):

        # The classes are sorted alphabetically for reproducibility reasons
        classes = sorted(self.classes)

        if not classes:
            return ""

        # Determine the required width of each column in the table
        largest_label_len = max(len(str(c)) for c in classes)
        largest_number_len = len(
            str(max(max(v for v in c.values()) for c in self.data.values()))
        )
        width = max(largest_label_len, largest_number_len) + 2

        # Make a template to print out rows one by one
        row_format = "{:>{width}}" * (len(classes) + 1)

        # Write down the header
        table = row_format.format("", *map(str, classes), width=width) + "\n"

        # Write down the true labels row by row
        table += "\n".join(
            (
                row_format.format(
                    str(y_true),
                    *[f"{self.data[y_true][y_pred]:0.0f}" for y_pred in classes],
                    width=width,
                )
                for y_true in classes
            )
        )

        return textwrap.dedent(table)

    def true_positives(self, label):
        return self.data[label][label]

    def true_negatives(self, label):
        return self.sum_diag - self.data[label][label]

    def false_positives(self, label):
        return self.sum_col[label] - self.data[label][label]

    def false_negatives(self, label):
        return self.sum_row[label] - self.data[label][label]
