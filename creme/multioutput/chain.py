import collections
import copy

from .. import base


class ClassifierChain(collections.OrderedDict, base.MultiOutputClassifier):
    """A multi-label model that arranges classifiers into a chain.

    Example:

        ::

            >>> from creme import feature_selection
            >>> from creme import linear_model
            >>> from creme import metrics
            >>> from creme import model_selection
            >>> from creme import multioutput
            >>> from creme import preprocessing
            >>> from creme import stream
            >>> from sklearn import datasets

            >>> X_y = stream.iter_sklearn_dataset(
            ...     dataset=datasets.fetch_openml('yeast', version=4),
            ...     shuffle=True,
            ...     random_state=42
            ... )

            >>> model = feature_selection.VarianceThreshold(threshold=0.01)
            >>> model |= preprocessing.StandardScaler()
            >>> model |= multioutput.ClassifierChain(
            ...     classifier=linear_model.LogisticRegression(),
            ...     order=list(range(14))
            ... )

            >>> metric = metrics.Jaccard()

            >>> for x, y in X_y:
            ...     # Convert y values to booleans
            ...     y = {i: yi == 'TRUE' for i, yi in y.items()}
            ...     y_pred = model.predict_one(x)
            ...     metric = metric.update(y, y_pred)
            ...     model = model.fit_one(x, y)

            >>> metric
            Jaccard: 0.458292

    """

    def __init__(self, classifier, order):
        super().__init__()
        for o in order:
            self[o] = copy.deepcopy(classifier)

    def fit_one(self, x, y):

        x = copy.copy(x)

        for o, clf in self.items():
            y_pred = clf.predict_one(x)
            clf.fit_one(x, y[o])
            x[o] = y_pred

        return self

    def predict_proba_one(self, x):

        x = copy.copy(x)
        y_pred = {}

        for o, clf in self.items():
            y_pred[o] = clf.predict_proba_one(x)
            x[o] = max(y_pred[o], key=y_pred[o].get)

        return y_pred
