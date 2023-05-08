from __future__ import annotations

import textwrap

from river import metrics


class MultiLabelConfusionMatrix:
    """Multi-label confusion matrix.

    Under the hood, this stores one `metrics.ConfusionMatrix` for each output.

    Examples
    --------

    >>> from river import metrics

    >>> cm = metrics.multioutput.MultiLabelConfusionMatrix()

    >>> y_true = [
    ...     {0: False, 1: True, 2: True},
    ...     {0: True, 1: True, 2: False}
    ... ]

    >>> y_pred = [
    ...     {0: True, 1: True, 2: True},
    ...     {0: True, 1: False, 2: False}
    ... ]

    >>> for yt, yp in zip(y_true, y_pred):
    ...     cm = cm.update(yt, yp)

    >>> cm
    0
                False   True
        False       0      1
         True       0      1
    <BLANKLINE>
    1
                False   True
        False       0      0
         True       1      1
    <BLANKLINE>
    2
                False   True
        False       1      0
         True       0      1

    """

    def __init__(self):
        self.data = dict()

    def update(self, y_true, y_pred, sample_weight=1.0):
        for label, yt in y_true.items():
            try:
                cm = self.data[label]
            except KeyError:
                cm = metrics.ConfusionMatrix()
                self.data[label] = cm
            cm.update(yt, y_pred[label], sample_weight)
        return self

    def revert(self, y_true, y_pred, sample_weight=1.0):
        for label, yt in y_true.items():
            try:
                cm = self.data[label]
            except KeyError:
                cm = metrics.ConfusionMatrix()
                self.data[label] = cm
            cm.update(yt, y_pred[label], sample_weight)
        return self

    def __repr__(self):
        return "\n\n".join(
            "\n".join([str(label)] + textwrap.indent(repr(cm), prefix="    ").splitlines())
            for label, cm in self.data.items()
        )
