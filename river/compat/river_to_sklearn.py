from __future__ import annotations

import copy
import typing

import numpy as np

try:
    import pandas as pd

    PANDAS_INSTALLED = True
except ImportError:
    PANDAS_INSTALLED = False
from sklearn import base as sklearn_base
from sklearn import pipeline, preprocessing, utils

from river import base, compose, stream

__all__ = [
    "convert_river_to_sklearn",
    "River2SKLRegressor",
    "River2SKLClassifier",
    "River2SKLClusterer",
    "River2SKLTransformer",
]


# Define a streaming method for each kind of batch input
STREAM_METHODS: dict[type, typing.Callable] = {np.ndarray: stream.iter_array}

if PANDAS_INSTALLED:
    STREAM_METHODS[pd.DataFrame] = stream.iter_pandas

# Params passed to sklearn.utils.check_X_y and sklearn.utils.check_array
SKLEARN_INPUT_X_PARAMS = {
    "accept_sparse": False,
    "accept_large_sparse": True,
    "dtype": "numeric",
    "order": None,
    "copy": False,
    "force_all_finite": True,
    "ensure_2d": True,
    "allow_nd": False,
    "ensure_min_samples": 1,
    "ensure_min_features": 1,
}

# Params passed to sklearn.utils.check_X_y in addition to SKLEARN_INPUT_X_PARAMS
SKLEARN_INPUT_Y_PARAMS = {"multi_output": False, "y_numeric": False}


def convert_river_to_sklearn(estimator: base.Estimator):
    """Wraps a river estimator to make it compatible with scikit-learn.

    Parameters
    ----------
    estimator

    """

    if isinstance(estimator, compose.Pipeline):
        return pipeline.Pipeline(
            [(name, convert_river_to_sklearn(step)) for name, step in estimator.steps.items()]
        )

    wrappers = [
        (base.Classifier, River2SKLClassifier),
        (base.Clusterer, River2SKLClusterer),
        (base.Regressor, River2SKLRegressor),
        (base.Transformer, River2SKLTransformer),
    ]

    for base_type, wrapper in wrappers:
        if isinstance(estimator, base_type):
            obj = wrapper(estimator)
            obj.instance_ = copy.deepcopy(estimator)
            return obj

    raise ValueError("Couldn't find an appropriate wrapper")


class River2SKLBase(base.Wrapper, sklearn_base.BaseEstimator):
    """This class is just here for house-keeping."""

    @property
    def _wrapped_model(self):
        return self.river_estimator

    _required_parameters = ["river_estimator"]


class River2SKLRegressor(River2SKLBase, sklearn_base.RegressorMixin):
    """Compatibility layer from River to scikit-learn for regression.

    Parameters
    ----------
    river_estimator

    """

    def __init__(self, river_estimator: base.Regressor):
        # Check the estimator is a Regressor
        if not isinstance(river_estimator, base.Regressor):
            raise ValueError("river_estimator is not a Regressor")

        self.river_estimator = river_estimator

    def _partial_fit(self, X, y):
        # Check the inputs
        X, y = utils.check_X_y(X, y, **SKLEARN_INPUT_X_PARAMS, **SKLEARN_INPUT_Y_PARAMS)

        # Store the number of features so that future inputs can be checked
        if hasattr(self, "n_features_in_") and X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}")
        self.n_features_in_ = X.shape[1]

        # scikit-learn's convention is that fit shouldn't mutate the input parameters; we have to
        # deep copy the provided estimator in order to respect this convention
        if not hasattr(self, "instance_"):
            self.instance_ = copy.deepcopy(self.river_estimator)

        # Call learn_one for each observation
        for x, yi in STREAM_METHODS[type(X)](X, y):
            self.instance_.learn_one(x, yi)

        return self

    def fit(self, X, y):
        """Fits to an entire dataset contained in memory.

        Parameters
        ----------
        X
            array-like of shape (n_samples, n_features).
        y
            array-like of shape n_samples.

        Returns
        -------
        self

        """

        # Reset the state if already fitted
        for attr in ("instance_", "n_features_in_"):
            self.__dict__.pop(attr, None)

        # Fit with one pass of the dataset
        return self._partial_fit(X, y)

    def partial_fit(self, X, y):
        """Fits incrementally on a portion of a dataset.

        Parameters
        ----------
        X
            array-like of shape (n_samples, n_features).
        y
            array-like of shape n_samples.

        Returns
        -------
        self

        """

        # Fit with one pass of the dataset portion
        return self._partial_fit(X, y)

    def predict(self, X) -> np.ndarray:
        """Predicts the target of an entire dataset contained in memory.

        Parameters
        ----------
        X
            array-like of shape (n_samples, n_features).

        Returns
        -------
        Predicted target values for each row of `X`.

        """

        # Check the fit method has been called
        utils.validation.check_is_fitted(self, attributes="instance_")

        # Check the input
        X = utils.check_array(X, **SKLEARN_INPUT_X_PARAMS)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}")

        # Make a prediction for each observation
        y_pred = np.empty(shape=len(X))
        for i, (x, _) in enumerate(stream.iter_array(X)):
            y_pred[i] = self.instance_.predict_one(x)

        return y_pred


class River2SKLClassifier(River2SKLBase, sklearn_base.ClassifierMixin):
    """Compatibility layer from River to scikit-learn for classification.

    Parameters
    ----------
    river_estimator

    """

    def __init__(self, river_estimator: base.Classifier):
        # Check the estimator is Classifier
        if not isinstance(river_estimator, base.Classifier):
            raise ValueError("estimator is not a Classifier")

        self.river_estimator = river_estimator

    def _more_tags(self):
        return {"binary_only": not self.river_estimator._multiclass}

    def _partial_fit(self, X, y, classes):
        # If first _partial_fit call, set the classes, else check consistency
        if not hasattr(self, "classes_"):
            self.classes_ = classes

        # Check the inputs
        X, y = utils.check_X_y(X, y, **SKLEARN_INPUT_X_PARAMS, **SKLEARN_INPUT_Y_PARAMS)

        # Check the number of classes agrees with the type of classifier
        if len(self.classes_) > 2 and not self.river_estimator._multiclass:
            # Only a warning for now so tests can pass, see scikit-learn issue
            # https://github.com/scikit-learn/scikit-learn/issues/16798#issuecomment-651784267
            # TODO: change to a ValueError when fixed
            import warnings

            warnings.warn(
                f"more than 2 classes were given but {self.river_estimator} is a"
                " binary classifier"
            )

        # Store the number of features so that future inputs can be checked
        if hasattr(self, "n_features_in_") and X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}")
        self.n_features_in_ = X.shape[1]

        # Check the target
        utils.multiclass.check_classification_targets(y)
        if set(y) - set(self.classes_):
            raise ValueError("classes should include all valid labels that can be in y")

        # scikit-learn's convention is that fit shouldn't mutate the input parameters; we have to
        # deep copy the provided estimator in order to respect this convention
        if not hasattr(self, "instance_"):
            self.instance_ = copy.deepcopy(self.river_estimator)

        # river's binary classifiers expects bools or 0/1 values
        if not self.river_estimator._multiclass:
            if not hasattr(self, "label_encoder_"):
                self.label_encoder_ = preprocessing.LabelEncoder().fit(self.classes_)
            y = self.label_encoder_.transform(y)

        # Call learn_one for each observation
        for x, yi in STREAM_METHODS[type(X)](X, y):
            self.instance_.learn_one(x, yi)

        return self

    def fit(self, X, y):
        """Fits to an entire dataset contained in memory.

        Parameters
        ----------
        X
            array-like of shape (n_samples, n_features).
        y
            array-like of shape n_samples.

        Returns
        -------
        self

        """

        # Reset the state if already fitted
        for attr in ("classes_", "instance_", "label_encoder_", "n_features_in_"):
            self.__dict__.pop(attr, None)

        # Fit with one pass of the dataset
        classes = utils.multiclass.unique_labels(y)
        return self._partial_fit(X, y, classes)

    def partial_fit(self, X, y, classes=None):
        """Fits incrementally on a portion of a dataset.

        Parameters
        ----------
        X
            array-like of shape (n_samples, n_features).
        y
            array-like of shape n_samples.
        classes
            Classes across all calls to partial_fit. This argument is required for the first call
            to partial_fit and can be omitted in the subsequent calls. Note that y doesn't need to
            contain all labels in `classes`.

        Returns
        -------
        self

        """

        # Fit with one pass of the dataset portion
        return self._partial_fit(X, y, classes)

    def predict_proba(self, X):
        """Predicts the target probability of an entire dataset contained in memory.

        Parameters
        ----------
        X
            array-like of shape (n_samples, n_features).

        Returns
        -------
        Predicted target values for each row of `X`.

        """

        # Check the fit method has been called
        utils.validation.check_is_fitted(self, attributes="instance_")

        # Check the input
        X = utils.check_array(X, **SKLEARN_INPUT_X_PARAMS)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}")

        # river's predictions have to converted to follow the scikit-learn conventions
        def reshape_probas(y_pred):
            return [y_pred.get(c, 0) for c in self.classes_]

        # Make a prediction for each observation
        y_pred = np.empty(shape=(len(X), len(self.classes_)))
        for i, (x, _) in enumerate(stream.iter_array(X)):
            y_pred[i] = reshape_probas(self.instance_.predict_proba_one(x))

        return y_pred

    def predict(self, X):
        """Predicts the target of an entire dataset contained in memory.

        Parameters
        ----------
        X
            array-like of shape (n_samples, n_features).

        Returns
        -------
        Predicted target values for each row of `X`.

        """

        # Check the fit method has been called
        utils.validation.check_is_fitted(self, attributes="instance_")

        # Check the input
        X = utils.check_array(X, **SKLEARN_INPUT_X_PARAMS)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}")

        # Make a prediction for each observation
        y_pred = [None] * len(X)
        for i, (x, _) in enumerate(stream.iter_array(X)):
            y_pred[i] = self.instance_.predict_one(x)

        # Convert back to the expected labels if an encoder was necessary for binary classification
        y_pred = np.asarray(y_pred)
        if hasattr(self, "label_encoder_"):
            y_pred = self.label_encoder_.inverse_transform(y_pred.astype(int))

        return y_pred


class River2SKLTransformer(River2SKLBase, sklearn_base.TransformerMixin):
    """Compatibility layer from River to scikit-learn for transformation.

    Parameters
    ----------
    river_estimator

    """

    def __init__(self, river_estimator: base.Transformer):
        # Check the estimator is a Transformer
        if not isinstance(river_estimator, base.Transformer):
            raise ValueError("estimator is not a Transformer")

        self.river_estimator = river_estimator

    def _partial_fit(self, X, y):
        # Check the inputs
        if y is None:
            X = utils.check_array(X, **SKLEARN_INPUT_X_PARAMS)
        else:
            X, y = utils.check_X_y(X, y, **SKLEARN_INPUT_X_PARAMS, **SKLEARN_INPUT_Y_PARAMS)

        # Store the number of features so that future inputs can be checked
        if hasattr(self, "n_features_in_") and X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}")
        self.n_features_in_ = X.shape[1]

        # scikit-learn's convention is that fit shouldn't mutate the input parameters; we have to
        # deep copy the provided estimator in order to respect this convention
        if not hasattr(self, "instance_"):
            self.instance_ = copy.deepcopy(self.river_estimator)

        # Call learn_one for each observation
        if isinstance(self.instance_, base.SupervisedTransformer):
            for x, yi in STREAM_METHODS[type(X)](X, y):
                self.instance_.learn_one(x, yi)
        else:
            for x, _ in STREAM_METHODS[type(X)](X):
                self.instance_.learn_one(x)

        return self

    def fit(self, X, y=None):
        """Fits to an entire dataset contained in memory.

        Parameters
        ----------
        X
            array-like of shape (n_samples, n_features).
        y
            array-like of shape n_samples.

        Returns
        -------
        self

        """

        # Reset the state if already fitted
        for attr in ("instance_", "n_features_in_"):
            self.__dict__.pop(attr, None)

        # Fit with one pass of the dataset
        return self._partial_fit(X, y)

    def partial_fit(self, X, y=None):
        """Fits incrementally on a portion of a dataset.

        Parameters
        ----------
        X
            array-like of shape (n_samples, n_features).
        y
            array-like of shape n_samples.

        Returns
        -------
        self

        """

        # Fit with one pass of the dataset
        return self._partial_fit(X, y)

    def transform(self, X):
        """Predicts the target of an entire dataset contained in memory.

        Parameters
        ----------
        X
            array-like of shape (n_samples, n_features)

        Returns
        -------
        Transformed output.

        """

        # Check the fit method has been called
        utils.validation.check_is_fitted(self, attributes="instance_")

        # Check the input
        X = utils.check_array(X, **SKLEARN_INPUT_X_PARAMS)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}")

        # Call predict_proba_one for each observation
        X_trans = [None] * len(X)
        for i, (x, _) in enumerate(STREAM_METHODS[type(X)](X)):
            X_trans[i] = list(self.instance_.transform_one(x).values())

        return np.asarray(X_trans)


class River2SKLClusterer(River2SKLBase, sklearn_base.ClusterMixin):
    """Compatibility layer from River to scikit-learn for clustering.

    Parameters
    ----------
    river_estimator

    """

    def __init__(self, river_estimator: base.Clusterer):
        # Check the estimator is a Clusterer
        if not isinstance(river_estimator, base.Clusterer):
            raise ValueError("estimator is not a Clusterer")

        self.river_estimator = river_estimator

    def _partial_fit(self, X, y):
        # Check the inputs
        X = utils.check_array(X, **SKLEARN_INPUT_X_PARAMS)

        # Store the number of features so that future inputs can be checked
        if hasattr(self, "n_features_in_") and X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}")
        self.n_features_in_ = X.shape[1]

        # scikit-learn's convention is that fit shouldn't mutate the input parameters; we have to
        # deep copy the provided estimator in order to respect this convention
        if not hasattr(self, "instance_"):
            self.instance_ = copy.deepcopy(self.river_estimator)

        # Call learn_one for each observation
        self.labels_ = np.empty(len(X), dtype=np.int32)
        for i, (x, _) in enumerate(STREAM_METHODS[type(X)](X)):
            label = self.instance_.learn_one(x).predict_one(x)
            self.labels_[i] = label

        return self

    def fit(self, X, y=None):
        """Fits to an entire dataset contained in memory.

        Parameters
        ----------
        X
            array-like of shape (n_samples, n_features).
        y
            array-like of shape n_samples.

        Returns
        -------
        self

        """

        # Reset the state if already fitted
        for attr in ("instance_", "n_features_in_"):
            self.__dict__.pop(attr, None)

        # Fit with one pass of the dataset
        return self._partial_fit(X, y)

    def partial_fit(self, X, y):
        """Fits incrementally on a portion of a dataset.

        Parameters
        ----------
        X
            array-like of shape (n_samples, n_features).
        y
            array-like of shape n_samples.

        Returns
        -------
        self

        """

        # Fit with one pass of the dataset
        return self._partial_fit(X, y)

    def predict(self, X):
        """Predicts the target of an entire dataset contained in memory.

        Parameters
        ----------
        X
            array-like of shape (n_samples, n_features).

        Returns
        -------
        Transformed output.

        """

        # Check the fit method has been called
        utils.validation.check_is_fitted(self, attributes="instance_")

        # Check the input
        X = utils.check_array(X, **SKLEARN_INPUT_X_PARAMS)

        # Call predict_proba_one for each observation
        y_pred = np.empty(len(X), dtype=np.int32)
        for i, (x, _) in enumerate(STREAM_METHODS[type(X)](X)):
            y_pred[i] = self.instance_.predict_one(x)

        return y_pred
