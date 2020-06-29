import copy
import typing

import numpy as np
try:
    import pandas as pd
    PANDAS_INSTALLED = True
except ImportError:
    PANDAS_INSTALLED = False
from sklearn import base as sklearn_base
from sklearn import pipeline
from sklearn import preprocessing
from sklearn import utils

from creme import base
from creme import compose
from creme import stream


__all__ = [
    'convert_creme_to_sklearn',
    'Creme2SKLRegressor',
    'Creme2SKLClassifier',
    'Creme2SKLClusterer',
    'Creme2SKLTransformer'
]


# Define a streaming method for each kind of batch input
STREAM_METHODS: typing.Dict[typing.Type, typing.Callable] = {
    np.ndarray: stream.iter_array
}

if PANDAS_INSTALLED:
    STREAM_METHODS[pd.DataFrame] = stream.iter_pandas

# Params passed to sklearn.utils.check_X_y and sklearn.utils.check_array
SKLEARN_INPUT_X_PARAMS = {
    'accept_sparse': False,
    'accept_large_sparse': True,
    'dtype': 'numeric',
    'order': None,
    'copy': False,
    'force_all_finite': True,
    'ensure_2d': True,
    'allow_nd': False,
    'ensure_min_samples': 1,
    'ensure_min_features': 1
}

# Params passed to sklearn.utils.check_X_y in addition to SKLEARN_INPUT_X_PARAMS
SKLEARN_INPUT_Y_PARAMS = {
    'multi_output': False,
    'y_numeric': False
}


def convert_creme_to_sklearn(estimator: base.Estimator):
    """Wraps a creme estimator to make it compatible with scikit-learn.

    Parameters:
        estimator

    """

    if isinstance(estimator, compose.Pipeline):
        return pipeline.Pipeline([
            (name, convert_creme_to_sklearn(step))
            for name, step in estimator.steps.items()
        ])

    wrappers = [
        (base.BinaryClassifier, Creme2SKLClassifier),
        (base.Clusterer, Creme2SKLClusterer),
        (base.MultiClassifier, Creme2SKLClassifier),
        (base.Regressor, Creme2SKLRegressor),
        (base.Transformer, Creme2SKLTransformer)
    ]

    for base_type, wrapper in wrappers:
        if isinstance(estimator, base_type):
            obj = wrapper(estimator)
            obj.instance_ = copy.deepcopy(estimator)
            return obj

    raise ValueError("Couldn't find an appropriate wrapper")


class Creme2SKLBase(sklearn_base.BaseEstimator):
    """This class is just here for house-keeping."""


class Creme2SKLRegressor(Creme2SKLBase, sklearn_base.RegressorMixin):
    """`creme` to `sklearn` regressor adapter.

    Parameters:
        estimator

    """

    def __init__(self, estimator: base.Regressor):
        self.estimator = estimator

    def fit(self, X, y):
        """Fits to an entire dataset contained in memory.

        Parameters:
            X (array-like of shape (n_samples, n_features))
            y (array-like of shape n_samples)

        Returns:
            self

        """

        # Check the estimator is a Regressor
        if not isinstance(self.estimator, base.Regressor):
            raise ValueError('estimator is not a Regressor')

        # Check the inputs
        X, y = utils.check_X_y(X, y, **SKLEARN_INPUT_X_PARAMS, **SKLEARN_INPUT_Y_PARAMS)

        # scikit-learn's convention is that fit shouldn't mutate the input parameters; we have to
        # deep copy the provided estimator in order to respect this convention
        self.instance_ = copy.deepcopy(self.estimator)

        # Call fit_one for each observation
        for x, yi in STREAM_METHODS[type(X)](X, y):
            self.instance_.fit_one(x, yi)

        return self

    def predict(self, X):
        """Predicts the target of an entire dataset contained in memory.

        Parameters:
            X (array-like of shape (n_samples, n_features))

        Returns:
            array of shape (n_samples,): Predicted target values for each row of `X`.

        """

        # Check the fit method has been called
        utils.validation.check_is_fitted(self, attributes='instance_')

        # Check the input
        X = utils.check_array(X, **SKLEARN_INPUT_X_PARAMS)

        # Make a prediction for each observation
        y_pred = np.empty(shape=len(X))
        for i, (x, _) in enumerate(stream.iter_array(X)):
            y_pred[i] = self.instance_.predict_one(x)

        return y_pred


class Creme2SKLClassifier(Creme2SKLBase, sklearn_base.ClassifierMixin):
    """`creme` to `sklearn` classification adapter.

    Parameters:
        estimator

    """

    def __init__(self, estimator: base.Classifier):
        self.estimator = estimator

    def fit(self, X, y):
        """Fits to an entire dataset contained in memory.

        Parameters:
            X (array-like of shape (n_samples, n_features))
            y (array-like of shape n_samples)

        Returns:
            self

        """

        # Check the estimator is either a BinaryClassifier or a MultiClassifier
        if not isinstance(self.estimator, (base.BinaryClassifier, base.MultiClassifier)):
            raise ValueError('estimator is not a BinaryClassifier nor a MultiClassifier')

        # Check the inputs
        X, y = utils.check_X_y(X, y, **SKLEARN_INPUT_X_PARAMS, **SKLEARN_INPUT_Y_PARAMS)

        # Check the target
        utils.multiclass.check_classification_targets(y)

        # Check the number of classes agrees with the type of classifier
        self.classes_ = np.unique(y)
        if len(self.classes_) > 2 and not isinstance(self.estimator, base.MultiClassifier):
            raise ValueError(f'n_classes is more than 2 but {self.estimator} is a ' +
                             'BinaryClassifier')

        # creme's BinaryClassifier expects bools or 0/1 values
        if not isinstance(self.estimator, base.MultiClassifier):
            self.label_encoder_ = preprocessing.LabelEncoder().fit(y)
            y = self.label_encoder_.transform(y)

        # scikit-learn's convention is that fit shouldn't mutate the input parameters; we have to
        # deep copy the provided estimator in order to respect this convention
        self.instance_ = copy.deepcopy(self.estimator)

        # Call fit_one for each observation
        for x, yi in STREAM_METHODS[type(X)](X, y):
            self.instance_.fit_one(x, yi)

        # Store the number of features so that future inputs can be checked
        self.n_features_ = X.shape[1]

        return self

    def predict_proba(self, X):
        """Predicts the target probability of an entire dataset contained in memory.

        Parameters:
            X (array-like of shape (n_samples, n_features))

        Returns:
            array of shape (n_samples,): Predicted target values for each row of `X`.

        """

        # Check the fit method has been called
        utils.validation.check_is_fitted(self, attributes='instance_')

        # Check the input
        X = utils.check_array(X, **SKLEARN_INPUT_X_PARAMS)

        if X.shape[1] != self.n_features_:
            raise ValueError(f'Expected {self.n_features_} features, got {X.shape[1]}')

        # creme's predictions have to converted to follow the scikit-learn conventions
        def reshape_probas(y_pred):
            return [y_pred.get(c, 0) for c in self.classes_]

        # Make a prediction for each observation
        y_pred = np.empty(shape=(len(X), len(self.classes_)))
        for i, (x, _) in enumerate(stream.iter_array(X)):
            y_pred[i] = reshape_probas(self.instance_.predict_proba_one(x))

        return y_pred

    def predict(self, X):
        """Predicts the target of an entire dataset contained in memory.

        Parameters:
            X (array-like of shape (n_samples, n_features))

        Returns:
            array of shape (n_samples,): Predicted target values for each row of `X`.

        """

        # Check the fit method has been called
        utils.validation.check_is_fitted(self, attributes='instance_')

        # Check the input
        X = utils.check_array(X, **SKLEARN_INPUT_X_PARAMS)

        if X.shape[1] != self.n_features_:
            raise ValueError(f'Expected {self.n_features_} features, got {X.shape[1]}')

        # Make a prediction for each observation
        y_pred = [None] * len(X)
        for i, (x, _) in enumerate(stream.iter_array(X)):
            y_pred[i] = self.instance_.predict_one(x)

        return np.asarray(y_pred)


class Creme2SKLTransformer(Creme2SKLBase, sklearn_base.TransformerMixin):
    """`creme` to `sklearn` transformer adapter.

    Parameters:
        estimator

    """

    def __init__(self, estimator: base.Transformer):
        self.estimator = estimator

    def fit(self, X, y=None):
        """Fits to an entire dataset contained in memory.

        Parameters:
            X (array-like of shape (n_samples, n_features))
            y (array-like of shape n_samples)

        Returns:
            self

        """

        # Check the estimator is a Transformer
        if not isinstance(self.estimator, base.Transformer):
            raise ValueError('estimator is not a Transformer')

        # Check the inputs
        if y is None:
            X = utils.check_array(X, **SKLEARN_INPUT_X_PARAMS)
        else:
            X, y = utils.check_X_y(X, y, **SKLEARN_INPUT_X_PARAMS, **SKLEARN_INPUT_Y_PARAMS)

        # scikit-learn's convention is that fit shouldn't mutate the input parameters; we have to
        # deep copy the provided estimator in order to respect this convention
        self.instance_ = copy.deepcopy(self.estimator)

        # Call fit_one for each observation
        if isinstance(self.instance_, base.SupervisedTransformer):
            for x, yi in STREAM_METHODS[type(X)](X, y):
                self.instance_.fit_one(x, yi)
        else:
            for x, _ in STREAM_METHODS[type(X)](X):
                self.instance_.fit_one(x)

        # Store the number of features so that future inputs can be checked
        self.n_features_ = X.shape[1]

        return self

    def transform(self, X):
        """Predicts the target of an entire dataset contained in memory.

        Parameters:
            X (array-like of shape (n_samples, n_features))

        Returns:
            array: Transformed output.

        """

        # Check the fit method has been called
        utils.validation.check_is_fitted(self, attributes='instance_')

        # Check the input
        X = utils.check_array(X, **SKLEARN_INPUT_X_PARAMS)

        if X.shape[1] != self.n_features_:
            raise ValueError(f'Expected {self.n_features_} features, got {X.shape[1]}')

        # Call predict_proba_one for each observation
        X_trans = [None] * len(X)
        for i, (x, _) in enumerate(STREAM_METHODS[type(X)](X)):
            X_trans[i] = list(self.instance_.transform_one(x).values())

        return np.asarray(X_trans)


class Creme2SKLClusterer(Creme2SKLBase, sklearn_base.ClusterMixin):
    """Wraps a `creme` clusterer to make it compatible with `sklearn`.

    Parameters:
        estimator

    """

    def __init__(self, estimator: base.Clusterer):
        self.estimator = estimator

    def fit(self, X, y=None):
        """Fits to an entire dataset contained in memory.

        Parameters:
            X (array-like of shape (n_samples, n_features))
            y (array-like of shape n_samples)

        Returns:
            self

        """

        # Check the estimator is a Clusterer
        if not isinstance(self.estimator, base.Clusterer):
            raise ValueError('estimator is not a Clusterer')

        # Check the inputs
        X = utils.check_array(X, **SKLEARN_INPUT_X_PARAMS)

        # scikit-learn's convention is that fit shouldn't mutate the input parameters; we have to
        # deep copy the provided estimator in order to respect this convention
        self.instance_ = copy.deepcopy(self.estimator)

        # Call fit_one for each observation
        self.labels_ = np.empty(len(X), dtype=np.int32)
        for i, (x, _) in enumerate(STREAM_METHODS[type(X)](X)):
            label = self.instance_.fit_one(x).predict_one(x)
            self.labels_[i] = label

        return self

    def predict(self, X):
        """Predicts the target of an entire dataset contained in memory.

        Parameters:
            X (array-like of shape (n_samples, n_features))

        Returns:
            array: Transformed output.

        """

        # Check the fit method has been called
        utils.validation.check_is_fitted(self, attributes='instance_')

        # Check the input
        X = utils.check_array(X, **SKLEARN_INPUT_X_PARAMS)

        # Call predict_proba_one for each observation
        y_pred = np.empty(len(X), dtype=np.int32)
        for i, (x, _) in enumerate(STREAM_METHODS[type(X)](X)):
            y_pred[i] = self.instance_.predict_one(x)

        return y_pred
