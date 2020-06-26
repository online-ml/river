import copy
import functools
import typing

import numpy as np
try:
    import pandas as pd
    PANDAS_INSTALLED = True
except ImportError:
    PANDAS_INSTALLED = False
from sklearn import base as sklearn_base
from sklearn import exceptions
from sklearn import pipeline
from sklearn import preprocessing
from sklearn import utils

from creme import base
from creme import compose
from creme import stream


__all__ = [
    'convert_creme_to_sklearn',
    'convert_sklearn_to_creme',
    'SKL2CremeClassifier',
    'SKL2CremeRegressor',
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


def convert_sklearn_to_creme(estimator: sklearn_base.BaseEstimator, batch_size=1,
                             classes: typing.Optional[list] = None):
    """Wraps a scikit-learn estimator to make it compatible with creme.

    Parameters:
        estimator
        batch_size: The amount of observations that will be used during each `partial_fit` call.
            Setting this to 1 means that the model will learn with each given observation.
            Increasing the batch size means that the observations will be stored in a buffer and
            the model will only update itself once enough observations are available.
        classes: Class names necessary for classifiers.

    """

    if not hasattr(estimator, 'partial_fit'):
        raise ValueError(f'{estimator} does not have a partial_fit method')

    wrappers = [
        (sklearn_base.RegressorMixin, functools.partial(
            SKL2CremeRegressor,
            batch_size=batch_size
        )),
        (sklearn_base.ClassifierMixin, functools.partial(
            SKL2CremeClassifier,
            batch_size=batch_size,
            classes=classes
        ))
    ]

    for base_type, wrapper in wrappers:
        if isinstance(estimator, base_type):
            return wrapper(copy.deepcopy(estimator))

    raise ValueError("Couldn't find an appropriate wrapper")


class SKL2CremeBase:

    def __init__(self, estimator: sklearn_base.BaseEstimator, batch_size: int,
                 x_dtype: typing.Type):
        self.estimator = estimator
        self.batch_size = batch_size
        self.x_dtype = x_dtype

        self._x_batch = None
        self._y_batch = None
        self._batch_i = 0


class SKL2CremeRegressor(SKL2CremeBase, base.Regressor):
    """`sklearn` to `creme` regressor adapter.

    Parameters:
        estimator: A scikit-learn transformer which has a `partial_fit` method.
        batch_size: The amount of observations that will be used during each `partial_fit` call.
            Setting this to 1 means that the model will learn with each given observation.
            Increasing the batch size means that the observations will be stored in a buffer and
            the model will only update itself once enough observations are available.

    Example:

        >>> from creme import compat
        >>> from creme import metrics
        >>> from creme import model_selection
        >>> from creme import preprocessing
        >>> from creme import stream
        >>> from sklearn import linear_model
        >>> from sklearn import datasets

        >>> X_y = stream.iter_sklearn_dataset(
        ...     dataset=datasets.load_boston(),
        ...     shuffle=True,
        ...     seed=42
        ... )

        >>> scaler = preprocessing.StandardScaler()
        >>> sgd_reg = compat.convert_sklearn_to_creme(linear_model.SGDRegressor())
        >>> model = scaler | sgd_reg

        >>> metric = metrics.MAE()

        >>> model_selection.progressive_val_score(X_y, model, metric)
        MAE: 11.004415

    """

    def __init__(self, estimator: sklearn_base.RegressorMixin, batch_size=1):
        super().__init__(
            estimator=estimator,
            batch_size=batch_size,
            x_dtype=np.float
        )

    def fit_one(self, x, y):

        if self._x_batch is None:
            n_features = len(x)
            self._x_batch = np.empty(shape=(self.batch_size, n_features), dtype=self.x_dtype)
            self._y_batch = [None] * self.batch_size

        self._x_batch[self._batch_i, :] = list(x.values())
        self._y_batch[self._batch_i] = y

        self._batch_i += 1

        if self._batch_i == self.batch_size:
            self.estimator.partial_fit(X=self._x_batch, y=self._y_batch)
            self._batch_i = 0

        return self

    def predict_one(self, x):
        try:
            return self.estimator.predict(X=[list(x.values())])[0]
        except exceptions.NotFittedError:
            return 0


class SKL2CremeClassifier(SKL2CremeBase, base.MultiClassifier):
    """`sklearn` to `creme` classifier adapter.

    Parameters:
        estimator: A scikit-learn regressor which has a `partial_fit` method.
        classes
        batch_size: The amount of observations that will be used during each `partial_fit` call.
            Setting this to 1 means that the model will learn with each given observation.
            Increasing the batch size means that the observations will be stored in a buffer and
            the model will only update itself once enough observations are available.

    Example:

        >>> from creme import compat
        >>> from creme import metrics
        >>> from creme import model_selection
        >>> from creme import preprocessing
        >>> from creme import stream
        >>> from sklearn import linear_model
        >>> from sklearn import datasets

        >>> X_y = stream.iter_sklearn_dataset(
        ...     dataset=datasets.load_breast_cancer(),
        ...     shuffle=True,
        ...     seed=42
        ... )

        >>> model = preprocessing.StandardScaler()
        >>> model |= compat.convert_sklearn_to_creme(
        ...     estimator=linear_model.SGDClassifier(
        ...         loss='log',
        ...         eta0=0.01,
        ...         learning_rate='constant'
        ...     ),
        ...     classes=[False, True]
        ... )

        >>> metric = metrics.LogLoss()

        >>> model_selection.progressive_val_score(X_y, model, metric)
        LogLoss: 0.199554

    """

    def __init__(self, estimator: sklearn_base.ClassifierMixin, classes: list, batch_size=1):
        super().__init__(
            estimator=estimator,
            batch_size=batch_size,
            x_dtype=np.float
        )
        self.classes = classes

    def fit_one(self, x, y):

        if self._x_batch is None:
            n_features = len(x)
            self._x_batch = np.empty(shape=(self.batch_size, n_features), dtype=self.x_dtype)
            self._y_batch = [None] * self.batch_size

        self._x_batch[self._batch_i, :] = list(x.values())
        self._y_batch[self._batch_i] = y

        self._batch_i += 1

        if self._batch_i == self.batch_size:
            self.estimator.partial_fit(X=self._x_batch, y=self._y_batch, classes=self.classes)
            self._batch_i = 0

        return self

    def predict_proba_one(self, x):
        try:
            y_pred = self.estimator.predict_proba([list(x.values())])[0]
            return {self.classes[i]: p for i, p in enumerate(y_pred)}
        except exceptions.NotFittedError:
            return {c: 1 / len(self.classes) for c in self.classes}

    def predict_one(self, x):
        try:
            y_pred = self.estimator.predict(X=[list(x.values())])[0]
            return y_pred
        except exceptions.NotFittedError:
            return self.classes[0]


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
