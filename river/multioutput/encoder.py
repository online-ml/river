from __future__ import annotations

import typing

from river import base


class MultiClassEncoder(base.MultiLabelClassifier):
    """Convert a multi-label task into multiclass.

    Assigns a class to each unique combination of labels, and proceeds
    with training the supplied multi-class classifier.

    The transformation is done by converting the label set, which could be seen
    as a binary number, into an integer representing a class. At prediction time,
    the predicted integer is converted back to a binary number which is the
    predicted label set.

    Parameters
    ----------
    model
        The classifier used for learning.

    Examples
    --------

    >>> from river import forest
    >>> from river import metrics
    >>> from river import multioutput
    >>> from river.datasets import synth

    >>> dataset = synth.Logical(seed=42, n_tiles=100)

    >>> model = multioutput.MultiClassEncoder(
    ...     model=forest.ARFClassifier(seed=7)
    ... )

    >>> metric = metrics.multioutput.MicroAverage(metrics.Jaccard())

    >>> for x, y in dataset:
    ...    y_pred = model.predict_one(x)
    ...    y_pred = {k: y_pred.get(k, 0) for k in y}
    ...    metric = metric.update(y, y_pred)
    ...    model = model.learn_one(x, y)

    >>> metric
    MicroAverage(Jaccard): 95.10%

    """

    def __init__(self, model: base.Classifier):
        super().__init__()
        self.model = model

        self._label_map: dict[tuple, int] = {}
        self._r_label_map: dict[int, tuple] = {}
        self._labels: set[typing.Hashable] = set()

    def learn_one(self, x, y):
        self._labels.update(y.keys())

        aux = tuple(sorted(y.items()))
        if aux not in self._label_map:
            code = len(self._label_map)
            self._label_map[aux] = code
            self._r_label_map[code] = aux

        # Encode
        y_encoded = self._label_map[aux]

        # Update the classifier
        self.model.learn_one(x, y_encoded)

        return self

    def predict_proba_one(self, x, **kwargs):
        enc_probas = self.model.predict_proba_one(x, **kwargs)

        if not enc_probas:
            return {}

        enc_class = max(enc_probas, key=enc_probas.get)

        return {
            label_id: {
                bool(label_val): enc_probas[enc_class],
                not bool(label_val): 1 - enc_probas[enc_class],
            }
            for label_id, label_val in self._r_label_map[enc_class]
        }

    @classmethod
    def _unit_test_params(cls):
        from river import tree

        yield {"model": tree.HoeffdingTreeClassifier()}
