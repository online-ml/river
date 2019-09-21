import collections
import copy

from .. import base


__all__ = ['ClassifierChain', 'RegressorChain']


class BaseChain(collections.OrderedDict):

    def __init__(self, model, order):
        super().__init__()
        for o in order:
            self[o] = copy.deepcopy(model)

    def fit_one(self, x, y):

        x = copy.copy(x)

        for o, clf in self.items():
            y_pred = clf.predict_one(x)
            clf.fit_one(x, y[o])
            x[o] = y_pred

        return self


class ClassifierChain(BaseChain, base.MultiOutputClassifier):
    """A multi-label model that arranges classifiers into a chain.

    Example:

        ::

            >>> from creme import feature_selection
            >>> from creme import linear_model
            >>> from creme import metrics
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
            ...     model=linear_model.LogisticRegression(),
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
            Jaccard: 0.450325

    """

    def predict_proba_one(self, x):

        x = copy.copy(x)
        y_pred = {}

        for o, clf in self.items():
            y_pred[o] = clf.predict_proba_one(x)
            x[o] = max(y_pred[o], key=y_pred[o].get)

        return y_pred


class RegressorChain(BaseChain, base.MultiOutputRegressor):
    """A multi-label model that arranges regressor into a chain.

    Example:

        ::

            >>> from creme import linear_model
            >>> from creme import metrics
            >>> from creme import model_selection
            >>> from creme import multioutput
            >>> from creme import preprocessing
            >>> from creme import stream
            >>> from sklearn import datasets

            >>> X_y = stream.iter_sklearn_dataset(
            ...     dataset=datasets.load_linnerud(),
            ...     shuffle=True,
            ...     random_state=42
            ... )

            >>> model = multioutput.RegressorChain(
            ...     model=(
            ...         preprocessing.StandardScaler() |
            ...         linear_model.LinearRegression(intercept_lr=0.3)
            ...     ),
            ...     order=[0, 1, 2]
            ... )

            >>> metric = metrics.RegressionMultiOutput(metrics.MAE())

            >>> model_selection.online_score(X_y, model, metric)
            MAE: 16.970895...

    """

    def predict_one(self, x):

        x = copy.copy(x)
        y_pred = {}

        for o, clf in self.items():
            y_pred[o] = clf.predict_one(x)
            x[o] = y_pred[o]

        return y_pred
