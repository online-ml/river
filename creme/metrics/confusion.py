import collections


class ConfusionMatrix(collections.defaultdict):
    """Confusion matrix.

    Calling ``print`` will pretty-print the confusion matrix.

    Example:

        >>> from creme import metrics

        >>> y_true = ['cat', 'ant', 'cat', 'cat', 'ant', 'bird']
        >>> y_pred = ['ant', 'ant', 'cat', 'cat', 'ant', 'cat']

        >>> cm = metrics.ConfusionMatrix()

        >>> for y_t, y_p in zip(y_true, y_pred):
        ...     cm = cm.update(y_t, y_p)

        >>> cm
                 ant  bird   cat
           ant     2     0     0
          bird     0     0     1
           cat     1     0     2

    """

    def __init__(self):
        super().__init__(collections.Counter)
        self.classes = set()

    def update(self, y_true, y_pred):
        self[y_true].update([y_pred])
        self.classes.update({y_true, y_pred})
        return self

    def __str__(self):

        # The classes are sorted alphabetically for reproducibility reasons
        classes = sorted(self.classes)

        # Determine the required width of each column in the table
        largest_label_len = max(map(len, classes))
        largest_number_len = len(str(max(max(counter.values()) for counter in self.values())))
        width = max(largest_label_len, largest_number_len) + 2

        # Make a template to print out rows one by one
        row_format = '{:>{width}}' * (len(classes) + 1)

        # Write down the header
        table = row_format.format('', *classes, width=width) + '\n'

        # Write down the true labels row by row
        table += '\n'.join((
            row_format.format(y_true, *[self[y_true][y_pred] for y_pred in classes], width=width)
            for y_true in classes
        ))

        return table

    def __repr__(self):
        return str(self)
