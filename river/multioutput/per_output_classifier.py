from __future__ import annotations

import collections
import copy

from river import base, linear_model

__all__ = ["PerOutputClassifier"]


class PerOutputClassifier(base.Wrapper, base.MultiLabelClassifier, collections.UserDict):  # type:ignore[misc]
    """A multi-output model that trains one independent classifier per output.

    This model does not use the prediction of one output as a feature for the next. Each output is
    modelled by its own copy of the base classifier, trained independently. (This is the streaming
    equivalent of scikit-learn's `MultiOutputClassifier`).

    The set of outputs isn't known from the start in a streaming setting, new classifiers are
    instantiated on the fly, one per output key encountered in `y`.

    Parameters
    ----------
    classifier
        A classifier model used for each label.

    Examples
    --------

    >>> import random

    >>> from river import datasets
    >>> from river import feature_selection
    >>> from river import linear_model
    >>> from river import metrics
    >>> from river import multioutput
    >>> from river import preprocessing

    >>> dataset = list(datasets.Yeast())
    >>> random.Random(42).shuffle(dataset)

    >>> model = feature_selection.VarianceThreshold(threshold=0.01)
    >>> model |= preprocessing.StandardScaler()
    >>> model |= multioutput.PerOutputClassifier(
    ...     classifier=linear_model.LogisticRegression(),
    ... )

    >>> metric = metrics.multioutput.MicroAverage(metrics.Jaccard())

    >>> for x, y in dataset:
    ...     y_pred = model.predict_one(x)
    ...     y_pred = {k: y_pred.get(k, False) for k in y}
    ...     metric.update(y, y_pred)
    ...     model.learn_one(x, y)

    >>> metric
    MicroAverage(Jaccard): 41.82%

    """

    def __init__(self, classifier: base.Classifier):
        super().__init__()
        self.classifier = classifier

    @property
    def _wrapped_model(self):
        return self.classifier

    def __getitem__(self, key):
        try:
            return collections.UserDict.__getitem__(self, key)
        except KeyError:
            collections.UserDict.__setitem__(self, key, copy.deepcopy(self.classifier))
            return self[key]

    @classmethod
    def _unit_test_params(cls):
        yield {"classifier": linear_model.LogisticRegression()}

    @property
    def _multiclass(self):
        return self.classifier._multiclass

    def learn_one(self, x, y, **kwargs):
        for o, y_o in y.items():
            self[o].learn_one(x, y_o, **kwargs)

    def predict_one(self, x, **kwargs):
        return {o: clf.predict_one(x, **kwargs) for o, clf in self.items()}

    def predict_proba_one(self, x, **kwargs):
        return {o: clf.predict_proba_one(x, **kwargs) for o, clf in self.items()}
