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
        decimals: The number of decimals to display in each cell.

    Example:

        >>> from creme import metrics

        >>> y_true = ['pear', 'apple', 'banana', 'banana', 'banana']
        >>> y_pred = ['apple', 'pear', 'banana', 'banana', 'apple']

        >>> report = metrics.ClassificationReport()

        >>> for yt, yp in zip(y_true, y_pred):
        ...     report = report.update(yt, yp)

        >>> print(report)
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

    .. tip::
        You can wrap a `creme.metrics.ClassificationReport` with `creme.metrics.Rolling` in order
        to obtain a classification report over a window of observations. You can also wrap it with
        `creme.metrics.TimeRolling` to obtain a report over a period of time.

    """

    def __init__(self, decimals=3):
        self.decimals = decimals
        self._f1s = collections.defaultdict(fbeta.F1)
        self._macro_precision = precision.MacroPrecision()
        self._macro_recall = recall.MacroRecall()
        self._macro_f1 = fbeta.MacroF1()
        self._micro_precision = precision.MicroPrecision()
        self._micro_recall = recall.MicroRecall()
        self._micro_f1 = fbeta.MicroF1()
        self._weighted_precision = precision.WeightedPrecision()
        self._weighted_recall = recall.WeightedRecall()
        self._weighted_f1 = fbeta.WeightedF1()
        self._accuracy = accuracy.Accuracy()
        self._support = collections.Counter()

    def get(self):
        raise NotImplementedError

    @property
    def bigger_is_better(self):
        return True

    @property
    def requires_labels(self):
        return True

    def update(self, y_true, y_pred, sample_weight=1.):

        self._support[y_true] += 1

        # Update per class metrics
        for c in self._support:
            self._f1s[c].update(y_true == c, y_pred == c, sample_weight)

        # Update global metrics
        for m in [self._macro_precision, self._macro_recall, self._macro_f1,
                  self._micro_precision, self._micro_recall, self._micro_f1,
                  self._weighted_precision, self._weighted_recall, self._weighted_f1]:
            m.update(y_true, y_pred, sample_weight)

        self._accuracy.update(y_true, y_pred)

        return self

    def revert(self, y_true, y_pred, sample_weight=1.):

        self._support[y_true] -= 1

        # Revert per class metrics
        for c in self._support:
            self._f1s[c].revert(y_true == c, y_pred == c, sample_weight)

        # Revert global metrics
        for m in [self._macro_precision, self._macro_recall, self._macro_f1,
                  self._micro_precision, self._micro_recall, self._micro_f1,
                  self._weighted_precision, self._weighted_recall, self._weighted_f1]:
            m.revert(y_true, y_pred, sample_weight)
        self._accuracy.revert(y_true, y_pred)

        return self

    def __repr__(self):

        def fmt_float(x):
            return f'{x:.{self.decimals}f}'

        headers = ['', 'Precision', 'Recall', 'F1', 'Support']
        classes = sorted(self._support.keys())
        columns = [
            # Row names
            ['', *map(str, classes), '', 'Macro', 'Micro', 'Weighted'],
            # Precision values
            [
                '', *[fmt_float(self._f1s[c].precision.get()) for c in classes], '',
                *map(fmt_float, [
                    self._macro_precision.get(),
                    self._micro_precision.get(),
                    self._weighted_precision.get()
                ])
            ],
            # Recall values
            [
                '', *[fmt_float(self._f1s[c].recall.get()) for c in classes], '',
                *map(fmt_float, [
                    self._macro_recall.get(),
                    self._micro_recall.get(),
                    self._weighted_recall.get()
                ])
            ],
            # F1 values
            [
                '', *[fmt_float(self._f1s[c].get()) for c in classes], '',
                *map(fmt_float, [
                    self._macro_f1.get(),
                    self._micro_f1.get(),
                    self._weighted_f1.get()
                ])
            ],
            # Support
            ['', *[str(self._support[c]) for c in classes], *[''] * 4]
        ]

        # Build the table
        table = utils.pretty.print_table(headers, columns)

        # Write down the accuracy
        width = len(table.splitlines()[0])
        accuracy = f'{self._accuracy.get():.{self.decimals - 2}%}' + ' accuracy'
        table += '\n\n' + f'{accuracy:^{width}}'

        return table
