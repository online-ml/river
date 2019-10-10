import collections


__all__ = ['ConfusionMatrix']


class ConfusionMatrix:
    """Confusion matrix.

    This class is different from the rest of the classes from the `metrics` module in that it
    doesn't have a ``get`` method.

    Attributes:
        classes (set): The entire set of seen classes, whether they are part of the predictions or
            the true values.

    Example:

        ::

            >>> from creme import metrics

            >>> y_true = ['cat', 'ant', 'cat', 'cat', 'ant', 'bird']
            >>> y_pred = ['ant', 'ant', 'cat', 'cat', 'ant', 'cat']

            >>> cm = metrics.ConfusionMatrix()

            >>> for y_t, y_p in zip(y_true, y_pred):
            ...     cm = cm.update(y_t, y_p)

            >>> cm
                     ant  bird   cat
               ant   2.0     0     0
              bird     0     0   1.0
               cat   1.0     0   2.0

            >>> cm['bird']['cat']
            1.0

    """

    def __init__(self):
        self.counts = collections.defaultdict(collections.Counter)
        self.class_counts = collections.Counter()

    def __getitem__(self, idx):
        """Syntactic sugar for accessing the counts directly."""
        return self.counts[idx]

    def update(self, y_true, y_pred, sample_weight=1.):
        self.counts[y_true].update({y_pred: sample_weight})
        self.class_counts.update([y_true, y_pred])
        return self

    def revert(self, y_true, y_pred, sample_weight=1.):
        self.counts[y_true].subtract({y_pred: sample_weight})
        self.class_counts.subtract([y_true, y_pred])
        return self

    @property
    def classes(self):
        return list(self.class_counts)

    def __str__(self):

        # The classes are sorted alphabetically for reproducibility reasons
        classes = sorted(self.classes)

        # Determine the required width of each column in the table
        largest_label_len = max(len(str(c)) for c in classes)
        largest_number_len = len(str(max(max(c.values()) for c in self.counts.values())))
        width = max(largest_label_len, largest_number_len) + 2

        # Make a template to print out rows one by one
        row_format = '{:>{width}}' * (len(classes) + 1)

        # Write down the header
        table = row_format.format('', *map(str, classes), width=width) + '\n'

        # Write down the true labels row by row
        table += '\n'.join((
            row_format.format(
                str(y_true),
                *[self.counts[y_true][y_pred] for y_pred in classes],
                width=width
            )
            for y_true in sorted(self.counts)
        ))

        return table

    def __repr__(self):
        return str(self)
