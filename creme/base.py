"""
Base classes used throughout the library.
"""

import abc

import numpy as np
try:
    import pandas as pd
    PANDAS_INSTALLED = True
except ImportError:
    PANDAS_INSTALLED = False
from sklearn import base
from sklearn import utils

from . import stream


STREAM_METHODS = {
    np.ndarray: stream.iter_numpy
}

if PANDAS_INSTALLED:
    STREAM_METHODS[pd.DataFrame] = stream.iter_pandas


class BaseEstimator(base.BaseEstimator):

    def get_params(self, deep=True):
        """Get parameters for this estimator.

        Parameters:
            deep (bool, optional) If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns:
            dict of string to any: Parameter names mapped to their values.

        """
        return super().get_params(deep=deep)

    def set_params(self, **params):
        """Set the parameters of this estimator.

        The method works on simple estimators as well as on nested objects
        (such as pipelines). The latter have parameters of the form
        ``<component>__<parameter>`` so that it's possible to update each
        component of a nested object.

        Returns:
            self

        """
        return super().set_params(**params)


class Regressor(BaseEstimator, base.RegressorMixin):

    @abc.abstractmethod
    def fit_one(x, y):
        """Fits to a set of features ``x`` and a real-valued target ``y``.

        Parameters:
            x (dict)
            y (float)

        Returns:
            float: The estimated target value of ``x`` before seeing ``y``.

        """
        pass

    def fit(self, X, y=None):
        """Fits to an entire dataset contained in memory.

        Parameters:
            X (array-like of shape (n_samples, n_features))
            y (array-like of shape n_samples)

        Returns:
            self

        """
        X = utils.check_array(X)
        for x, yi in STREAM_METHODS[type(X)](X, y):
            self.fit_one(x, yi)
        return self

    @abc.abstractmethod
    def predict_one(x: dict) -> float:
        """Predicts the target value of a set of features ``x``

        Parameters:
            x (dict)

        Returns:
            float

        """
        pass

    def predict(self, X):
        """Predicts the target of an entire dataset contained in memory.

        Parameters:
            X (array-like of shape (n_samples, n_features))

        Returns:
            array of shape (n_samples,): Predicted target values for each row of ``X``.

        """
        X = utils.check_array(X)
        y_pred = np.empty(len(X), dtype='float')
        for i, (x, _) in enumerate(stream.iter_numpy(X, shuffle=False)):
            y_pred[i] = self.predict_one(x)
        return y_pred

    def fit_predict(self, X, y):
        """Fits and predicts an entire dataset contained in memory.

        Parameters:
            X (array-like of shape (n_samples, n_features))
            y (array-like of shape (n_samples,))

        Returns:
            array of shape (n_samples,): Predicted target values for each row of ``X``.

        """
        return super().fit_predict(X, y)

    def score(self, X, y, sample_weight=None):
        """Returns the coefficient of determination R^2 of the prediction.

        The coefficient R^2 is defined as (1 - u/v), where u is the residual sum of squares
        ((y_true - y_pred) ** 2).sum() and v is the total sum of squares
        ((y_true - y_true.mean()) ** 2).sum(). The best possible score is 1.0 and it can be
        negative (because the model can be arbitrarily worse). A constant model that always
        predicts the expected value of y, disregarding the input features, would get a R^2 score of
        0.0.

        Parameters:

            X (array-like of shape (n_samples, n_features)): Test samples. For some estimators this
            may be a precomputed kernel matrix instead, shape = (n_samples, n_samples_fitted),
            where n_samples_fitted is the number of samples used in the fitting for the estimator.
            y (array-like of shape = (n_samples) or (n_samples, n_outputs): True values for X.
            sample_weight (array-like of shape = [n_samples], optional): Sample weights.

        Returns:
            float: R^2 of self.predict(X) w.r.t. y.

        """
        return super().score(X, y, sample_weight)


class BinaryClassifier(BaseEstimator, base.ClassifierMixin):

    @abc.abstractmethod
    def fit_one(x, y):
        """Fits to a set of features ``x`` and a boolean target ``y``.

        Parameters:
            x (dict)
            y (bool)

        Returns:
            float: The estimated probability of ``x`` before seeing ``y``.

        """
        pass

    def fit(self, X, y=None):
        """Fits to an entire dataset contained in memory.

        Parameters:
            X (array-like of shape (n_samples, n_features))
            y (array-like of shape n_samples)

        Returns:
            self

        """
        X = utils.check_array(X)
        for x, yi in STREAM_METHODS[type(X)](X, y):
            self.fit_one(x, yi)
        return self

    @abc.abstractmethod
    def predict_proba_one(x: dict) -> float:
        """Predicts the probability output of a set of features ``x``

        Parameters:
            x (dict)

        Returns:
            float

        """
        pass

    @abc.abstractmethod
    def predict_one(x: dict) -> bool:
        """Predicts the target value of a set of features ``x``

        Parameters:
            x (dict)

        Returns:
            bool

        """
        pass

    def predict_proba(self, X):
        """Predicts the target probability of an entire dataset contained in memory.

        Parameters:
            X (array-like of shape (n_samples, n_features))

        Returns:
            array of shape (n_samples,): Predicted target values for each row of ``X``.

        """
        X = utils.check_array(X)
        y_pred = np.empty(len(X))
        for i, (x, _) in enumerate(stream.iter_numpy(X, shuffle=False)):
            y_pred[i] = self.predict_proba_one(x)
        return y_pred

    def predict(self, X):
        """Predicts the target of an entire dataset contained in memory.

        Parameters:
            X (array-like of shape (n_samples, n_features))

        Returns:
            array of shape (n_samples,): Predicted target values for each row of ``X``.

        """
        X = utils.check_array(X)
        return np.round(self.predict_proba(X)).astype(bool)

    def fit_predict(self, X, y):
        """Fits and predicts an entire dataset contained in memory.

        Parameters:
            X (array-like of shape (n_samples, n_features))
            y (array-like of shape (n_samples,))

        Returns:
            array of shape (n_samples,): Predicted target values for each row of ``X``.

        """
        return super().fit_predict(X, y)

    def score(self, X, y, sample_weight=None):
        """Returns the mean accuracy on the given test data and labels.

        Parameters:

            X (array-like of shape (n_samples, n_features)): Test samples.
            y (array-like of shape = (n_samples) or (n_samples, n_outputs): True labels.

        Returns:
            float: Mean accuracy of self.predict(X) w.r.t. y.

        """
        return super().score(X, y, sample_weight)


class MultiClassifier(BaseEstimator, base.ClassifierMixin):

    @abc.abstractmethod
    def fit_one(x, y):
        """Fits to a set of features ``x`` and a string target ``y``.

        Parameters:
            x (dict)
            y (str)

        Returns:
            float: The estimated class probabilities of ``x`` before seeing ``y``.

        """
        pass

    def fit(self, X, y=None):
        """Fits to an entire dataset contained in memory.

        Parameters:
            X (array-like of shape (n_samples, n_features))
            y (array-like of shape n_samples)

        Returns:
            self

        """
        X = utils.check_array(X)
        for x, yi in STREAM_METHODS[type(X)](X, y):
            self.fit_one(x, yi)
        return self

    @abc.abstractmethod
    def predict_proba_one(x: dict) -> dict:
        """Predicts the class probabilities of a set of features ``x``

        Parameters:
            x (dict)

        Returns:
            dict: A ``dict`` mapping classes to probabilities.

        """
        pass

    @abc.abstractmethod
    def predict_one(x: dict) -> str:
        """Predicts the class of a set of features ``x``

        Parameters:
            x (dict)

        Returns:
            str

        """
        pass

    def predict_proba(self, X):
        """Predicts the target of an entire dataset contained in memory.

        Parameters:
            X (array-like of shape (n_samples, n_features))

        Returns:
            array of shape (n_samples,): Predicted target values for each row of ``X``.

        """
        X = utils.check_array(X)
        y_pred = [None] * len(X)
        for i, (x, y) in enumerate(stream.iter_numpy(X, shuffle=False)):
            y_pred[i] = list(self.fit_one(x, y).values())
        return np.array(y_pred)

    def predict(self, X):
        """Predicts the target of an entire dataset contained in memory.

        Parameters:
            X (array-like of shape (n_samples, n_features))

        Returns:
            array of shape (n_samples,): Predicted target values for each row of ``X``.

        """
        X = utils.check_array(X)
        y_pred = [None] * len(X)
        for i, (x, _) in enumerate(stream.iter_numpy(X, shuffle=False)):
            y_pred[i] = self.predict_one(x)
        return np.array(y_pred)

    def fit_predict(self, X, y):
        """Fits and predicts an entire dataset contained in memory.

        Parameters:
            X (array-like of shape (n_samples, n_features))
            y (array-like of shape (n_samples,))

        Returns:
            array of shape (n_samples,): Predicted target values for each row of ``X``.

        """
        return super().fit_predict(X, y)

    def score(self, X, y, sample_weight=None):
        """Returns the mean accuracy on the given test data and labels.

        In multi-label classification, this is the subset accuracy which is a harsh metric since
        you require for each sample that each label set be correctly predicted.

        Parameters:

            X (array-like of shape (n_samples, n_features)): Test samples.
            y (array-like of shape = (n_samples) or (n_samples, n_outputs): True labels.

        Returns:
            float: Mean accuracy of self.predict(X) w.r.t. y.

        """
        return super().score(X, y, sample_weight)


class Transformer(BaseEstimator, base.TransformerMixin):

    @abc.abstractmethod
    def fit_one(x, y):
        """Fits to a set of features ``x ` and an optinal target ``y``.

        Parameters:
            x (dict)
            y

        Returns:
            dict: The transformation applied to ``x``.

        """
        pass

    def fit(self, X, y=None):
        """Fits to an entire dataset contained in memory.

        Parameters:
            X (array-like of shape (n_samples, n_features))
            y (array-like of shape n_samples)

        Returns:
            self

        """
        X = utils.check_array(X)
        for x, yi in STREAM_METHODS[type(X)](X, y):
            self.fit_one(x, yi)
        return self

    @abc.abstractmethod
    def transform_one(x: dict) -> dict:
        """Transformes a set of features ``x``

        Parameters:
            x (dict)

        Returns:
            dict

        """
        pass

    def transform(self, X):
        """Transforms an entire dataset contained in memory.

        Parameters:
            X (array-like of shape (n_samples, n_features))

        Returns:
            array: The transformed dataset.

        """
        X = utils.check_array(X)
        y_trans = [None] * len(X)
        for i, (x, _) in enumerate(stream.iter_numpy(X)):
            y_trans[i] = list(self.transform_one(x).values())
        return np.array(y_trans)

    def fit_transform(self, X, y):
        """Fits and transforms an entire dataset contained in memory.

        Parameters:
            X (array-like of shape (n_samples, n_features))
            y (array-like of shape (n_samples,))

        Returns:
            array: The transformed dataset.

        """
        return super().fit_predict(X, y)


class Clusterer(BaseEstimator, base.ClusterMixin):

    @abc.abstractmethod
    def fit_one(x, y=None):
        """Fits to a set of features ``x``.

        Parameters:
            x (dict)
            y: Not used, present here for API consistency by convention.

        Returns:
            int: The number of the cluster ``x`` belongs to.

        """
        pass

    def fit(self, X, y=None):
        """Fits to an entire dataset contained in memory.

        Parameters:
            X (array-like of shape (n_samples, n_features))
            y (array-like of shape n_samples)

        Returns:
            self

        """
        X = utils.check_array(X)
        for x, yi in STREAM_METHODS[type(X)](X, y):
            self.fit_one(x, yi)
        return self

    @abc.abstractmethod
    def predict_one(x: dict) -> dict:
        """Predicts the cluster number of a set of features ``x``

        Parameters:
            x (dict)

        Returns:
            int

        """
        pass

    def predict(self, X):
        """Predicts an entire dataset contained in memory.

        Parameters:
            X (array-like of shape (n_samples, n_features))

        Returns:
            array of shape (n_samples,)

        """
        X = utils.check_array(X)
        y_pred = np.empty(len(X), dtype='int32')
        for i, (x, _) in enumerate(stream.iter_numpy(X, shuffle=False)):
            y_pred[i] = self.predict_one(x)
        return y_pred

    def fit_predict(self, X, y):
        """Fits and predicts an entire dataset contained in memory.

        Parameters:
            X (array-like of shape (n_samples, n_features))
            y (array-like of shape (n_samples,))

        Returns:
            array of shape (n_samples,): Predicted cluster numbers for each row of ``X``.

        """
        return super().fit_predict(X, y)
