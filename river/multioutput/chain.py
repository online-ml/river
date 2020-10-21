import collections
import copy
import random

from river import base
from river import linear_model


__all__ = ['ClassifierChain', 'RegressorChain']


class BaseChain(base.WrapperMixin, collections.UserDict):

    def __init__(self, model, order=None, seed=None):
        super().__init__()
        self.model = model
        self.order = order
        self.seed = seed

        # If the order is specified, then we can instantiate a model for each label, if not we'll
        # do it in the first call to learn_one
        if isinstance(order, list):
            self._init_models()

    @property
    def _wrapped_model(self):
        return self.model

    def _init_models(self):
        for o in self.order:
            self[o] = copy.deepcopy(self.model)

    def _check_order(self, y):
        if self.order is None:
            self.order = list(y.keys())
            self._init_models()
        elif self.order == 'random':
            random.seed(self.seed)
            self.order = list(y.keys())
            random.shuffle(self.order)
            self._init_models()


class ClassifierChain(BaseChain, base.Classifier, base.MultiOutputMixin):
    """A multi-output model that arranges classifiers into a chain.

    This will create one model per output. The prediction of the first output will be used as a
    feature in the second output. The prediction for the second output will be used as a feature
    for the third, etc. This "chain model" is therefore capable of capturing dependencies between
    outputs.

    Parameters
    ----------
    model
    order
        A list with the targets order in which to construct the chain.
        If `None` then the order will be inferred from the order of the keys in the first
        provided target dictionary.
        If `'random'` then the order is set at random.
    seed
        Random number generator seed for reproducibility. Only relevant in `order='random'`


    Examples
    --------

    >>> from river import feature_selection
    >>> from river import linear_model
    >>> from river import metrics
    >>> from river import multioutput
    >>> from river import preprocessing
    >>> from river import stream
    >>> from sklearn import datasets

    >>> dataset = stream.iter_sklearn_dataset(
    ...     dataset=datasets.fetch_openml('yeast', version=4),
    ...     shuffle=True,
    ...     seed=42
    ... )

    >>> model = feature_selection.VarianceThreshold(threshold=0.01)
    >>> model |= preprocessing.StandardScaler()
    >>> model |= multioutput.ClassifierChain(
    ...     model=linear_model.LogisticRegression(),
    ...     order=list(range(14))
    ... )

    >>> metric = metrics.Jaccard()

    >>> for x, y in dataset:
    ...     # Convert y values to booleans
    ...     y = {i: yi == 'TRUE' for i, yi in y.items()}
    ...     y_pred = model.predict_one(x)
    ...     metric = metric.update(y, y_pred)
    ...     model = model.learn_one(x, y)

    >>> metric
    Jaccard: 0.451524

    References
    ----------
    [^1]: [Multi-Output Chain Models and their Application in Data Streams](https://jmread.github.io/talks/2019_03_08-Imperial_Stats_Seminar.pdf)

    """

    def __init__(self, model: base.Classifier, order: list = None, seed: int = None):
        super().__init__(model, order, seed)

    @classmethod
    def _default_params(cls):
        return {'model': linear_model.LogisticRegression()}

    def _multiclass(self):
        return self.model._multiclass

    def learn_one(self, x, y):

        self._check_order(y)

        x = copy.copy(x)

        for o in self.order:
            clf = self[o]

            # Make predictions before the model is updated to avoid leakage
            y_pred = clf.predict_proba_one(x)

            clf.learn_one(x, y[o])

            # The predictions are stored as features for the next label
            if clf._multiclass:
                for label, proba in y_pred.items():
                    x[f'{o}_{label}'] = proba
            else:
                x[o] = y_pred[True]

        return self

    def predict_proba_one(self, x):

        x = copy.copy(x)
        y_pred = {}

        if not isinstance(self.order, list):
            return y_pred

        for o in self.order:
            clf = self[o]

            y_pred[o] = clf.predict_proba_one(x)

            # The predictions are stored as features for the next label
            if clf._multiclass:
                for label, proba in y_pred.items():
                    x[f'{o}_{label}'] = proba
            else:
                x[o] = y_pred[o][True]

        return y_pred

    def predict_one(self, x):
        y_pred = self.predict_proba_one(x)
        return {
            c: max(y_pred[c], key=y_pred[c].get)
            for c in y_pred
        }


class RegressorChain(BaseChain, base.Regressor, base.MultiOutputMixin):
    """A multi-output model that arranges regressor into a chain.

    This will create one model per output. The prediction of the first output will be used as a
    feature in the second output. The prediction for the second output will be used as a feature
    for the third, etc. This "chain model" is therefore capable of capturing dependencies between
    outputs.

    Parameters
    ----------
    model
    order
        A list with the targets order in which to construct the chain.
        If `None` then the order will be inferred from the order of the keys in the first
        provided target dictionary.
        If `'random'` then the order is set at random.

    Examples
    --------

    >>> from river import evaluate
    >>> from river import linear_model
    >>> from river import metrics
    >>> from river import multioutput
    >>> from river import preprocessing
    >>> from river import stream
    >>> from sklearn import datasets

    >>> dataset = stream.iter_sklearn_dataset(
    ...     dataset=datasets.load_linnerud(),
    ...     shuffle=True,
    ...     seed=42
    ... )

    >>> model = multioutput.RegressorChain(
    ...     model=(
    ...         preprocessing.StandardScaler() |
    ...         linear_model.LinearRegression(intercept_lr=0.3)
    ...     ),
    ...     order=[0, 1, 2]
    ... )

    >>> metric = metrics.RegressionMultiOutput(metrics.MAE())

    >>> evaluate.progressive_val_score(dataset, model, metric)
    MAE: 16.396347

    """

    def __init__(self, model: base.Regressor, order: list = None, seed: int = None):
        super().__init__(model, order, seed)

    @classmethod
    def _default_params(cls):
        return {'model': linear_model.LinearRegression()}

    def learn_one(self, x, y):

        self._check_order(y)

        x = copy.copy(x)

        for o in self.order:
            reg = self[o]

            # Make predictions before the model is updated to avoid leakage
            y_pred = reg.predict_one(x)

            reg.learn_one(x, y[o])

            # The predictions are stored as features for the next label
            x[o] = y_pred

        return self

    def predict_one(self, x):

        x = copy.copy(x)
        y_pred = {}

        if not isinstance(self.order, list):
            return y_pred

        for o, clf in self.items():
            y_pred[o] = clf.predict_one(x)
            x[o] = y_pred[o]

        return y_pred
