import warnings

import numpy as np
cimport numpy as np

from collections import defaultdict


cdef class ConfusionMatrix:
    """ Confusion Matrix for binary-class and multi-class classification.

    Parameters
    ----------
    classes: set, list (default=None)
        The initial set of classes.

    Notes
    -----
    This confusion matrix is a 2D matrix of shape ``(n_classes, n_classes)``, corresponding
    to a single-target (binary and multi-class) classification task.

    Each row represents ``true`` (actual) class-labels, while each column corresponds
    to the ``predicted`` class-labels. For example, an entry in position ``[1, 2]`` means
    that the true class-label is 1, and the predicted class-label is 2 (incorrect prediction).

    This structure is used to keep updated statistics about a single-output classifier's
    performance and to compute multiple evaluation metrics.

    Warnings
    --------
    Implementation assumes zero-based contiguous class-labels.

    """
    _fmt = '0.0f'

    def __init__(self, classes=None):
        self._init_classes = set(classes) if classes is not None else set()
        self.classes = self._init_classes
        self.sum_diag = 0.0
        self.sum_row = defaultdict(float)
        self.sum_col = defaultdict(float)
        self.data = defaultdict(lambda: defaultdict(float))

    def __setitem__(self, entry, double sample_weight):
        if len(entry) != 2:
            raise KeyError(f'Expected (true_idx, pred_idx) entry, received: {entry}')
        y_true, y_pred = entry
        self.update(y_true, y_pred, sample_weight)

    def __getitem__(self, key):
        """Syntactic sugar for accessing the counts directly."""
        return self.data[key]

    def update(self, y_true, y_pred, sample_weight=1.):
        if sample_weight is None:
            # Since we ca not set a default value in the signature
            sample_weight = 1.0

        self.classes.update([y_true, y_pred])
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


cdef class MultiLabelConfusionMatrix:
    """ Multi-label Confusion Matrix.

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
    n_labels: int, optional, (default=2)
        The number of (initial) labels.

    Warnings
    --------
    Implementation assumes zero-based contiguous class-labels.

    """
    def __init__(self, int n_labels=2):
        if n_labels < 2:
            warnings.warn('n_labels can not be less than 2, using default (2)')
            n_labels = 2
        self._init_n_labels = n_labels
        self.labels = set(range(n_labels))
        self.n_labels = n_labels
        self.data = np.zeros((self.n_labels, 2, 2))
        self._max_label = max(self.labels)

    def __setitem__(self, key, double sample_weight):
        if not isinstance(key, tuple):
            raise KeyError('Expected (label, y_true, y_pred) tuple, received: {}'.format(type(key)))
        if sample_weight is None:
            # Since we ca not set a default value in the signature
            sample_weight = 1.0

        cdef int label = key[0]
        cdef int y_true = key[1]
        cdef int y_pred = key[2]

        if not (0 <= y_true < 2) or not (0 <= y_pred < 2):
            raise ValueError("Valid values are 0 or 1, passed ({}, {})".format(y_true, y_pred))

        if label > self._max_label:
            self._max_label = label
            # Extend the labels set assuming zero-based contiguous label values
            self.labels = set(range(self._max_label + 1))
            self.n_labels = self._max_label + 1
            self._reshape()

        self.data[label, y_true, y_pred] += sample_weight

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            raise KeyError('Expected (label, y_true, y_pred) tuple, received: {}'.format(type(key)))
        return self.data[key[0], key[1], key[2]]

    cdef void _reshape(self):
        current_n_labels = self.data.shape[0]
        if self._max_label + 1 > current_n_labels:
            n_labels_to_add = self._max_label + 1 - current_n_labels
            self.data = np.vstack((self.data, np.zeros((n_labels_to_add, 2, 2))))

    @property
    def shape(self):
        return self.data.shape

    def reset(self):
        self.__init__(n_labels=self._init_n_labels)

    def __str__(self):
        return self.data.__str__()