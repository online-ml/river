import collections
import functools
import warnings

import numpy as np
cimport numpy as np


cdef class ConfusionMatrix:
    """Confusion Matrix for binary-class and multi-class classification.

    Parameters
    ----------
    classes: set, list (default=None)
        The initial set of classes.

    Examples
    --------

    >>> from creme import metrics

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
    This confusion matrix is a 2D matrix of shape ``(n_classes, n_classes)``, corresponding
    to a single-target (binary and multi-class) classification task.

    Each row represents ``true`` (actual) class-labels, while each column corresponds
    to the ``predicted`` class-labels. For example, an entry in position ``[1, 2]`` means
    that the true class-label is 1, and the predicted class-label is 2 (incorrect prediction).

    This structure is used to keep updated statistics about a single-output classifier's
    performance and to compute multiple evaluation metrics.

    """

    _fmt = '0.0f'

    def __init__(self, classes=None):
        self._init_classes = set(classes) if classes is not None else set()
        self.sum_diag = 0.0
        self.sum_row = collections.defaultdict(float)
        self.sum_col = collections.defaultdict(float)
        self.data = collections.defaultdict(functools.partial(collections.defaultdict, float))
        self.n_samples = 0

    def __getitem__(self, key):
        """Syntactic sugar for accessing the counts directly."""
        return self.data[key]

    def update(self, y_true, y_pred, sample_weight=1.):
        self.n_samples += sample_weight
        self.data[y_true][y_pred] += sample_weight

        if y_true == y_pred:
            self.sum_diag += sample_weight
        self.sum_row[y_true] += sample_weight
        self.sum_col[y_pred] += sample_weight

        return self

    def revert(self, y_true, y_pred, sample_weight=1.):
        # Revert is equal to subtracting so we pass the negative sample_weight
        self.update(y_true, y_pred, -sample_weight)
        return self

    @property
    def classes(self):
        return (
            set(c for c, n in self.sum_row.items() if n) |
            set(c for c, n in self.sum_col.items() if n)
        )

    @property
    def n_classes(self):
        return len(self.classes)

    @property
    def shape(self):
        return self.n_classes, self.n_classes

    def reset(self):
        self.__init__(classes=self._init_classes)

    def __repr__(self):

        # The classes are sorted alphabetically for reproducibility reasons
        classes = sorted(self.classes)

        if not classes:
            return ''

        # Determine the required width of each column in the table
        largest_label_len = max(len(str(c)) for c in classes)
        largest_number_len = len(str(max(max(v for v in c.values()) for c in self.data.values())))
        width = max(largest_label_len, largest_number_len) + 2

        # Make a template to print out rows one by one
        row_format = '{:>{width}}' * (len(classes) + 1)

        # Write down the header
        table = row_format.format('', *map(str, classes), width=width) + '\n'

        # Write down the true labels row by row
        table += '\n'.join((
            row_format.format(
                str(y_true),
                *[f'{self.data[y_true][y_pred]:{self._fmt}}' for y_pred in classes],
                width=width
            )
            for y_true in classes
        ))

        return table

    def true_positives(self, label):
        return self.data[label][label]

    def true_negatives(self, label):
        return self.sum_diag - self.data[label][label]

    def false_positives(self, label):
        return self.sum_col[label] - self.data[label][label]

    def false_negatives(self, label):
        return self.sum_row[label] - self.data[label][label]


cdef class MultiLabelConfusionMatrix:
    """Multi-label Confusion Matrix.

    Notes
    -----
    This confusion matrix corresponds to a 3D matrix of shape ``(n_labels, 2, 2)`` meaning
    that each ``label`` has a corresponding binary ``(2x2)`` confusion matrix.

    The first dimension corresponds to the ``label``, the second and third dimensions
    are binary indicators for the ``true`` (actual) vs ``predicted`` values. For example,
    an entry in position ``[2, 0, 1]`` represents a miss-classification of label 2.

    This structure is used to keep updated statistics about a multi-output classifier's
    performance and to compute multiple evaluation metrics.

    Parameters
    ----------
    labels: set, list, optional, (default=None)
        The set of (initial) labels.

    """
    def __init__(self, labels=None):
        self._init_labels = set() if labels is None else set(labels)
        self.labels = self._init_labels
        self.n_labels = len(self.labels)
        if self.n_labels > 2:
            self.data = np.zeros((self.n_labels, 2, 2))
        else:
            # default to 2 labels
            self.data = np.zeros((2, 2, 2))
        self._label_dict = dict()
        self._label_idx_cnt = 0
        for label in self.labels:
            self._add_label(label)

    def update(self, label, int y_true, int y_pred, double sample_weight=1.0):
        if not (0 <= y_true < 2) or not (0 <= y_pred < 2):
            raise ValueError("Valid values are 0 or 1, passed ({}, {})".format(y_true, y_pred))

        # Increase sample count, negative sample_weight indicates that we are removing samples
        self.n_samples += 1 if sample_weight > 0. else -1

        label_idx = self._map_label(label, add_label=True)
        if label_idx > self.data.shape[0]:
            self._reshape()
        self.data[label_idx, y_true, y_pred] += sample_weight
        return self

    def revert(self, label, int y_true, int y_pred, sample_weight=1.):
        # Revert is equal to subtracting so we pass the negative sample_weight
        self.update(label, y_true, y_pred, -sample_weight)
        return self

    def __getitem__(self, label):
        if label in self.labels:
            label_idx = self._map_label(label, add_label=False)
            return self.data[label_idx]
        raise KeyError(f'Unknown label: {label}')

    cdef int _map_label(self, label, bint add_label):
        try:
            label_key = self._label_dict[label]
        except KeyError:
            if add_label:
                self._add_label(label)
                label_key = self._label_dict[label]
            else:
                label_key = -1
                raise KeyError(f'Unknown label: {label}')
        return label_key

    cdef void _add_label(self, label):
        self._label_dict[label] = self._label_idx_cnt
        if self._label_idx_cnt > self.data.shape[0] - 1:
            self._reshape()
        self._label_idx_cnt += 1
        self.labels.add(label)
        self.n_labels = len(self.labels)

    cdef void _reshape(self):
        self.data = np.vstack((self.data, np.zeros((1, 2, 2))))

    @property
    def shape(self):
        return self.data.shape

    def reset(self):
        self.__init__(labels=self._init_labels)

    def __str__(self):
        return self.data.__str__()

    def __repr__(self):
        # The labels are sorted alphabetically for reproducibility reasons
        labels = sorted(self.labels)

        if not labels:
            return  ''

        # Determine the required width of each column in the table
        largest_label_len = max(len(str(label)) for label in labels)
        largest_value_len = len(str(self.data[:].max()))
        width = max(5, largest_label_len, largest_value_len) + 2   # Min value is 5=len('label')

        # Make a template to print out rows one by one
        row_format = '{:>{width}}' * 5    # Label, TP, FP, FN, TN

        # Write down the header
        table = row_format.format('Label', 'TP', 'FP', 'FN', 'TN', width=width) + '\n'

        # Write down the values per labels row by row
        for label in labels:
            label_idx = self._map_label(label, add_label=False)
            table += ''.join(
                row_format.format(
                    str(label),                         # Label
                    self.data[label_idx][1][1],         # TP
                    self.data[label_idx][0][1],         # FP
                    self.data[label_idx][1][0],         # FN
                    self.data[label_idx][0][0],         # TN
                    width=width))
            table += '\n'

        return table
