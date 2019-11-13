import collections

from . import base
from . import accuracy
from . import fbeta
from . import precision
from . import recall


class ClassificationReport(base.MultiClassMetric):
    """A report for monitoring a classifier.

    This class maintains a set of metrics and updates each of them every time `update` is called.
    You can print this class at any time during a model's lifetime to get a tabular visualization
    of various metrics.

    Parameters:
        digits (int): The number of decimals to display for each metric.

    Attributes:
        f1s (collections.defaultdict): Contains an instance of `metrics.F1` for each label.
        macro_precision (metrics.MacroPrecision)
        macro_recall (metrics.MacroRecall)
        macro_f1 (metrics.MacroF1)
        micro_precision (metrics.MicroPrecision)
        micro_recall (metrics.MicroRecall)
        micro_f1 (metrics.MicroF1)
        weighted_precision (metrics.WeightedPrecision)
        weighted_recall (metrics.WeightedRecall)
        weighted_f1 (metrics.WeightedF1)
        accuracy (metrics.Accuracy)
        support (collections.Counter): Records the number of times a label is encountered.

    Example:

        ::

            >>> from creme import metrics

            >>> y_true = ['pear', 'apple', 'banana', 'banana', 'banana']
            >>> y_pred = ['apple', 'pear', 'banana', 'banana', 'apple']

            >>> report = ClassificationReport()

            >>> for yt, yp in zip(y_true, y_pred):
            ...     report = report.update(yt, yp)

            >>> print(report)
                     Precision   Recall       F1  Support
            <BLANKLINE>
                apple    0.000    0.000    0.000        1
               banana    1.000    0.667    0.800        3
                 pear    0.000    0.000    0.000        1
            <BLANKLINE>
                Macro    0.333    0.222    0.267
                Micro    0.400    0.400    0.400
             Weighted    0.600    0.400    0.480
            <BLANKLINE>
                              40.0% accuracy

    Note:
        You can wrap a `metrics.ClassificationReport` with `metrics.Rolling` in order to obtain
        a classification report over a recent window of observations.

    """

    def __init__(self, digits=3):
        self.digits = digits
        self.f1s = collections.defaultdict(fbeta.F1)
        self.macro_precision = precision.MacroPrecision()
        self.macro_recall = recall.MacroRecall()
        self.macro_f1 = fbeta.MacroF1()
        self.micro_precision = precision.MicroPrecision()
        self.micro_recall = recall.MicroRecall()
        self.micro_f1 = fbeta.MicroF1()
        self.weighted_precision = precision.WeightedPrecision()
        self.weighted_recall = recall.WeightedRecall()
        self.weighted_f1 = fbeta.WeightedF1()
        self.accuracy = accuracy.Accuracy()
        self.support = collections.Counter()

    @property
    def bigger_is_better(self):
        return True

    @property
    def get(self):
        raise NotImplementedError

    @property
    def requires_labels(self):
        return True

    def update(self, y_true, y_pred, sample_weight=1.):

        self.support[y_true] += 1

        for c in self.support:
            self.f1s[c].update(y_true == c, y_pred == c, sample_weight)

        for m in [self.macro_precision, self.macro_recall, self.macro_f1,
                  self.micro_precision, self.micro_recall, self.micro_f1,
                  self.weighted_precision, self.weighted_recall, self.weighted_f1]:
            m.update(y_true, y_pred, sample_weight)

        self.accuracy.update(y_true, y_pred)

        return self

    def revert(self, y_true, y_pred, sample_weight=1.):

        self.support[y_true] -= 1

        for c in self.support:
            self.f1s[c].revert(y_true == c, y_pred == c, sample_weight)

        for m in [self.macro_precision, self.macro_recall, self.macro_f1,
                  self.micro_precision, self.micro_recall, self.micro_f1,
                  self.weighted_precision, self.weighted_recall, self.weighted_f1]:
            m.revert(y_true, y_pred, sample_weight)

        self.accuracy.revert(y_true, y_pred)

        return self

    def __repr__(self):

        # The classes are sorted alphabetically for reproducibility reasons
        classes = sorted(self.support.keys())
        headers = ['Precision', 'Recall', 'F1', 'Support']

        # Determine the required width of each column in the table
        largest_header_len = max(len(str(h)) for h in classes)
        width = max(largest_header_len, self.digits + 2) + 3

        # Make a template to print out rows one by one
        row_format = '{:>{width}}' * (len(headers) + 1)

        # Write down the header
        table = row_format.format('', *map(str, headers), width=width) + '\n\n'

        # Write down the precision, recall, F1 scores along with the support for each class
        table += '\n'.join((
            row_format.format(
                str(c),
                *[
                    f'{self.f1s[c].precision.get():.{self.digits}f}',
                    f'{self.f1s[c].recall.get():.{self.digits}f}',
                    f'{self.f1s[c].get():.{self.digits}f}',
                    f'{self.support[c]}'
                ],
                width=width
            )
            for c in classes
        ))

        # Write down the macro, micro, and weighted metrics
        table += '\n\n' + '\n'.join([
            row_format.format(
                'Macro',
                *[
                    f'{self.macro_precision.get():.{self.digits}f}',
                    f'{self.macro_recall.get():.{self.digits}f}',
                    f'{self.macro_f1.get():.{self.digits}f}',
                    ''
                ],
                width=width
            ),
            row_format.format(
                'Micro',
                *[
                    f'{self.micro_precision.get():.{self.digits}f}',
                    f'{self.micro_recall.get():.{self.digits}f}',
                    f'{self.micro_f1.get():.{self.digits}f}',
                    ''
                ],
                width=width
            ),
            row_format.format(
                'Weighted',
                *[
                    f'{self.weighted_precision.get():.{self.digits}f}',
                    f'{self.weighted_recall.get():.{self.digits}f}',
                    f'{self.weighted_f1.get():.{self.digits}f}',
                    ''
                ],
                width=width
            )
        ])

        # Write down the accuracy
        table += '\n\n' + ' ' * (width * 2) + f'{self.accuracy.get():.{self.digits - 2}%} accuracy'

        return table
