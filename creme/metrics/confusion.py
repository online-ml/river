import collections
import functools
import operator


__all__ = ['ConfusionMatrix', 'RollingConfusionMatrix']


class ConfusionMatrix(collections.defaultdict):
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
               ant     2     0     0
              bird     0     0     1
               cat     1     0     2

            >>> cm['bird']['cat']
            1

    """

    def __init__(self):
        super().__init__(collections.Counter)
        self.classes_ = set()

    def update(self, y_true, y_pred):
        self[y_true].update([y_pred])
        self.classes_.update({y_true, y_pred})
        return self

    @property
    def classes(self):
        return self.classes_

    def __str__(self):

        # The classes are sorted alphabetically for reproducibility reasons
        classes = sorted(self.classes)

        # Determine the required width of each column in the table
        largest_label_len = max(len(str(c)) for c in classes)
        largest_number_len = len(str(max(max(counter.values()) for counter in self.values())))
        width = max(largest_label_len, largest_number_len) + 2

        # Make a template to print out rows one by one
        row_format = '{:>{width}}' * (len(classes) + 1)

        # Write down the header
        table = row_format.format('', *classes, width=width) + '\n'

        # Write down the true labels row by row
        table += '\n'.join((
            row_format.format(y_true, *[self[y_true][y_pred] for y_pred in classes], width=width)
            for y_true in sorted(self)
        ))

        return table

    def __repr__(self):
        return str(self)


class RollingConfusionMatrix(ConfusionMatrix):
    """Rolling confusion matrix.

    This class is different from the rest of the classes from the `metrics` module in that it
    doesn't have a ``get`` method.

    Parameters:
        window_size (int): The size of the window of most recent values to consider.

    Attributes:
        classes (set): The entire set of seen classes, whether they are part of the predictions or
            the true values.

    Example:

        ::

            >>> from creme import metrics

            >>> y_true = [0, 1, 2, 2, 2]
            >>> y_pred = [0, 0, 2, 2, 1]


            >>> cm = metrics.RollingConfusionMatrix(window_size=3)

            >>> for y_t, y_p in zip(y_true, y_pred):
            ...     cm = cm.update(y_t, y_p)
            ...     print(cm)
            ...     print('-' * 13)
                 0
              0  1
            -------------
                 0  1
              0  1  0
              1  1  0
            -------------
                 0  1  2
              0  1  0  0
              1  1  0  0
              2  0  0  1
            -------------
                 0  1  2
              1  1  0  0
              2  0  0  2
            -------------
                 1  2
              2  1  2
            -------------

    """

    def __init__(self, window_size):
        super().__init__()
        self.events = collections.deque(maxlen=window_size)

    @property
    def window_size(self):
        return self.events.maxlen

    def update(self, y_true, y_pred):

        # Update the appropriate counter
        self[y_true].update([y_pred])

        # If the events window is full then decrement the appropriate counter
        if len(self.events) == self.events.maxlen:
            yt, yp = self.events[0][0], self.events[0][1]
            self[yt].subtract([yp])

            # Remove empty counters
            if not self[yt][yp]:
                del self[yt][yp]
            if not self[yt]:
                del self[yt]

        self.events.append((y_true, y_pred))
        return self

    @property
    def classes(self):
        return functools.reduce(
            operator.or_,
            [set(self[yt].keys()) for yt in self] + [set(self.keys())]
        )
