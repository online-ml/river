import copy

import pandas as pd

from river import base, linear_model

__all__ = ["OneVsRestClassifier"]


class OneVsRestClassifier(base.WrapperMixin, base.Classifier):
    """One-vs-the-rest (OvR) multiclass strategy.

    This strategy consists in fitting one binary classifier per class. Because we are in a
    streaming context, the number of classes isn't known from the start. Hence, new classifiers are
    instantiated on the fly. Likewise, the predicted probabilities will only include the classes
    seen up to a given point in time.

    Note that this classifier supports mini-batches as well as single instances.

    The computational complexity for both learning and predicting grows linearly with the number of
    classes. If you have a very large number of classes, then you might want to consider using an
    `multiclass.OutputCodeClassifier` instead.

    Parameters
    ----------
    classifier
        A binary classifier, although a multi-class classifier will work too.

    Attributes
    ----------
    classifiers : dict
        A mapping between classes and classifiers.

    Examples
    --------

    >>> from river import datasets
    >>> from river import evaluate
    >>> from river import linear_model
    >>> from river import metrics
    >>> from river import multiclass
    >>> from river import preprocessing

    >>> dataset = datasets.ImageSegments()

    >>> scaler = preprocessing.StandardScaler()
    >>> ovr = multiclass.OneVsRestClassifier(linear_model.LogisticRegression())
    >>> model = scaler | ovr

    >>> metric = metrics.MacroF1()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    MacroF1: 0.774573

    This estimator also also supports mini-batching.

    >>> for X in pd.read_csv(dataset.path, chunksize=64):
    ...     y = X.pop('category')
    ...     y_pred = model.predict_many(X)
    ...     model = model.learn_many(X, y)

    """

    def __init__(self, classifier: base.Classifier):
        self.classifier = classifier
        self.classifiers = {}
        self._y_name = None

    @property
    def _wrapped_model(self):
        return self.classifier

    @property
    def _multiclass(self):
        return True

    @classmethod
    def _unit_test_params(cls):
        return {"classifier": linear_model.LogisticRegression()}

    def learn_one(self, x, y):

        # Instantiate a new binary classifier if the class is new
        if y not in self.classifiers:
            self.classifiers[y] = copy.deepcopy(self.classifier)

        # Train each label's associated classifier
        for label, model in self.classifiers.items():
            model.learn_one(x, bool(y == label))

        return self

    def predict_proba_one(self, x):

        y_pred = {}
        total = 0.0

        for label, model in self.classifiers.items():
            yp = model.predict_proba_one(x)[True]
            y_pred[label] = yp
            total += yp

        if total:
            return {label: votes / total for label, votes in y_pred.items()}
        return {label: 1 / len(y_pred) for label in y_pred}

    def learn_many(self, X, y, **params):

        self._y_name = y.name

        # Instantiate a new binary classifier for the classes that have not yet been seen
        for label in y.unique():
            if label not in self.classifiers:
                self.classifiers[label] = copy.deepcopy(self.classifier)

        # Train each label's associated classifier
        for label, model in self.classifiers.items():
            model.learn_many(X, y == label, **params)

        return self

    def predict_proba_many(self, X):

        y_pred = pd.DataFrame(columns=self.classifiers.keys(), index=X.index)

        for label, clf in self.classifiers.items():
            y_pred[label] = clf.predict_proba_many(X)[True]

        return y_pred.div(y_pred.sum(axis="columns"), axis="rows")

    def predict_many(self, X):
        if not self.classifiers:
            return pd.Series([None] * len(X), index=X.index, dtype="object")
        return self.predict_proba_many(X).idxmax(axis="columns").rename(self._y_name)
