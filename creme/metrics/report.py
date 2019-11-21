import collections

from .. import utils

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

            >>> report = metrics.ClassificationReport()

            >>> for yt, yp in zip(y_true, y_pred):
            ...     report = report.update(yt, yp)

            >>> print(report)
            ... # doctest: +NORMALIZE_WHITESPACE
                       Precision   Recall   F1      Support
            <BLANKLINE>
               apple       0.000    0.000   0.000         1
              banana       1.000    0.667   0.800         3
                pear       0.000    0.000   0.000         1
            <BLANKLINE>
               Macro       0.333    0.222   0.267
               Micro       0.400    0.400   0.400
            Weighted       0.600    0.400   0.480
            <BLANKLINE>
                             40.0% accuracy

    Note:
        You can wrap a `metrics.ClassificationReport` with `metrics.Rolling` in order to obtain
        a classification report over a recent window of observations.

    """

    def __init__(self, decimals=3):
        self.decimals = decimals
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

        # Update per class metrics
        for c in self.support:
            self.f1s[c].update(y_true == c, y_pred == c, sample_weight)

        # Update global metrics
        for m in [self.macro_precision, self.macro_recall, self.macro_f1,
                  self.micro_precision, self.micro_recall, self.micro_f1,
                  self.weighted_precision, self.weighted_recall, self.weighted_f1]:
            m.update(y_true, y_pred, sample_weight)

        self.accuracy.update(y_true, y_pred)

        return self

    def revert(self, y_true, y_pred, sample_weight=1.):

        self.support[y_true] -= 1

        # Revert per class metrics
        for c in self.support:
            self.f1s[c].revert(y_true == c, y_pred == c, sample_weight)

        # Revert global metrics
        for m in [self.macro_precision, self.macro_recall, self.macro_f1,
                  self.micro_precision, self.micro_recall, self.micro_f1,
                  self.weighted_precision, self.weighted_recall, self.weighted_f1]:
            m.revert(y_true, y_pred, sample_weight)
        self.accuracy.revert(y_true, y_pred)

        return self

    def __repr__(self):

        def fmt_float(x):
            return f'{x:.{self.decimals}f}'

        headers = ['', 'Precision', 'Recall', 'F1', 'Support']
        classes = sorted(self.support.keys())
        columns = [
            # Row names
            ['', *classes, '', 'Macro', 'Micro', 'Weighted'],
            # Precision values
            [
                '', *[fmt_float(self.f1s[c].precision.get()) for c in classes], '',
                *map(fmt_float, [
                    self.macro_precision.get(),
                    self.micro_precision.get(),
                    self.weighted_precision.get()
                ])
            ],
            # Recall values
            [
                '', *[fmt_float(self.f1s[c].recall.get()) for c in classes], '',
                *map(fmt_float, [
                    self.macro_recall.get(),
                    self.micro_recall.get(),
                    self.weighted_recall.get()
                ])
            ],
            # F1 values
            [
                '', *[fmt_float(self.f1s[c].get()) for c in classes], '',
                *map(fmt_float, [
                    self.macro_f1.get(),
                    self.micro_f1.get(),
                    self.weighted_f1.get()
                ])
            ],
            # Support
            ['', *[str(self.support[c]) for c in classes], *[''] * 4]
        ]

        # Build the table
        table = utils.pretty.print_table(headers, columns)

        # Write down the accuracy
        width = len(table.splitlines()[0])
        accuracy = f'{self.accuracy.get():.{self.decimals - 2}%}' + ' accuracy'
        table += '\n\n' + f'{accuracy:^{width}}'

        return table
