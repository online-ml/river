import typing

from river import base, utils


class MulticlassEncoder(base.MultiLabelClassifier):
    """Convert a multi-label task into multiclass.

    For each unique combination of labels, assigns a class and proceeds
    with training the supplied model.

    The transformation is done by changing the label set which could be seen
    as a binary number to an int which will represent the class, and after
    the prediction the int is converted back to a binary number which is the
    predicted label-set.

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

    >>> model = multioutput.MulticlassEncoder(
    ...     model=forest.ARFClassifier(seed=7)
    ... )

    >>> metric = metrics.multioutput.MicroAverage(metrics.Jaccard())

    >>> for x, y in dataset:
    ...    y_pred = model.predict_one(x)
    ...    metric = metric.update(y, y_pred)
    ...    model = model.learn_one(x, y)

    >>> metric
    MicroAverage(Jaccard): 95.41%

    """

    def __init__(self, model: base.Classifier):
        super().__init__()
        self.model = model

        self._next_code: int = 0
        self._label_map: typing.Dict[typing.Tuple, int] = {}
        self._r_label_map: typing.Dict[int, typing.Tuple] = {}
        self._labels: typing.Set[typing.Hashable] = set()

    def learn_one(self, x, y):
        self._labels.update(y.keys())

        aux = tuple(sorted(y.items()))
        if aux not in self._label_map:
            # Direct and reverse mapping
            self._label_map[aux] = self._next_code
            self._r_label_map[self._next_code] = aux
            # Update code
            self._next_code += 1

        # Encode
        y_encoded = self._label_map[aux]

        # Update the model
        self.model.learn_one(x, y_encoded)

        return self

    def predict_proba_one(self, x, **kwargs):
        enc_probas = self.model.predict_proba_one(x, **kwargs)

        if not enc_probas:
            return {}

        enc_class = max(enc_probas, key=enc_probas.get)

        result = {label: {False: 0.0, True: 0.0} for label in self._labels}

        for label_id, label_val in self._r_label_map[enc_class]:
            result[label_id][label_val] = enc_probas[enc_class]
            result[label_id] = utils.norm.normalize_values_in_dict(result[label_id])

        return result

    @classmethod
    def _unit_test_params(cls):
        from river import tree

        yield {"model": tree.HoeffdingTreeClassifier()}
