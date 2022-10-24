from river import metrics, utils


class ClassificationReport(metrics.base.MultiClassMetric):
    """A report for monitoring a classifier.

    This class maintains a set of metrics and updates each of them every time `update` is called.
    You can print this class at any time during a model's lifetime to get a tabular visualization
    of various metrics.

    You can wrap a `metrics.ClassificationReport` with `utils.Rolling` in order to obtain a
    classification report over a window of observations. You can also wrap it with
    `utils.TimeRolling` to obtain a report over a period of time.

    Parameters
    ----------
    decimals
        The number of decimals to display in each cell.
    cm
        This parameter allows sharing the same confusion
        matrix between multiple metrics. Sharing a confusion matrix reduces the amount of storage
        and computation time.

    Examples
    --------

    >>> from river import metrics

    >>> y_true = ['pear', 'apple', 'banana', 'banana', 'banana']
    >>> y_pred = ['apple', 'pear', 'banana', 'banana', 'apple']

    >>> report = metrics.ClassificationReport()

    >>> for yt, yp in zip(y_true, y_pred):
    ...     report = report.update(yt, yp)

    >>> print(report)
                   Precision   Recall   F1       Support
    <BLANKLINE>
       apple       0.00%    0.00%    0.00%         1
      banana     100.00%   66.67%   80.00%         3
        pear       0.00%    0.00%    0.00%         1
    <BLANKLINE>
       Macro      33.33%   22.22%   26.67%
       Micro      40.00%   40.00%   40.00%
    Weighted      60.00%   40.00%   48.00%
    <BLANKLINE>
                     40.00% accuracy

    """

    def __init__(self, decimals=2, cm=None):
        super().__init__(cm)
        self.decimals = decimals
        self._f1s = {}
        self._macro_precision = metrics.MacroPrecision(self.cm)
        self._macro_recall = metrics.MacroRecall(self.cm)
        self._macro_f1 = metrics.MacroF1(self.cm)
        self._micro_precision = metrics.MicroPrecision(self.cm)
        self._micro_recall = metrics.MicroRecall(self.cm)
        self._micro_f1 = metrics.MicroF1(self.cm)
        self._weighted_precision = metrics.WeightedPrecision(self.cm)
        self._weighted_recall = metrics.WeightedRecall(self.cm)
        self._weighted_f1 = metrics.WeightedF1(self.cm)
        self._accuracy = metrics.Accuracy(self.cm)

    def get(self):
        raise NotImplementedError

    def __repr__(self):
        def fmt_pct(x):
            return f"{x:.{self.decimals}%}"

        headers = ["", "Precision", "Recall", "F1", "Support"]
        classes = sorted(self.cm.classes)

        for c in classes:
            if c not in self._f1s:
                self._f1s[c] = metrics.F1(cm=self.cm, pos_val=c)

        columns = [
            # Row names
            ["", *map(str, classes), "", "Macro", "Micro", "Weighted"],
            # Precision values
            [
                "",
                *[fmt_pct(self._f1s[c].precision.get()) for c in classes],
                "",
                *map(
                    fmt_pct,
                    [
                        self._macro_precision.get(),
                        self._micro_precision.get(),
                        self._weighted_precision.get(),
                    ],
                ),
            ],
            # Recall values
            [
                "",
                *[fmt_pct(self._f1s[c].recall.get()) for c in classes],
                "",
                *map(
                    fmt_pct,
                    [
                        self._macro_recall.get(),
                        self._micro_recall.get(),
                        self._weighted_recall.get(),
                    ],
                ),
            ],
            # F1 values
            [
                "",
                *[fmt_pct(self._f1s[c].get()) for c in classes],
                "",
                *map(
                    fmt_pct,
                    [
                        self._macro_f1.get(),
                        self._micro_f1.get(),
                        self._weighted_f1.get(),
                    ],
                ),
            ],
            # Support
            ["", *[str(self.cm.sum_row[c]).rstrip("0").rstrip(".") for c in classes], *[""] * 4],
        ]

        # Build the table
        table = utils.pretty.print_table(headers, columns)

        # Write down the accuracy
        width = len(table.splitlines()[0])
        accuracy = f"{self._accuracy.get():.{self.decimals}%}" + " accuracy"
        table += "\n\n" + f"{accuracy:^{width}}"

        return table
