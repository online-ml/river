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


class Regressor(base.BaseEstimator, base.RegressorMixin):

    @abc.abstractmethod
    def fit_one(x, y):
        """Fits to set of features `x` and a real-valued target `y` and returns the estimated
        target value of `x` before seeing `y`.
        """
        pass

    def fit(self, X, y=None):
        X = utils.check_array(X)
        for x, yi in STREAM_METHODS[type(X)](X, y):
            self.fit_one(x, yi)
        return self

    @abc.abstractmethod
    def predict_one(x: dict) -> float:
        pass

    def predict(self, X):
        X = utils.check_array(X)
        y_pred = np.empty(len(X), dtype='float')
        for i, (x, _) in enumerate(stream.iter_numpy(X, shuffle=False)):
            y_pred[i] = self.predict_one(x)
        return y_pred


class BinaryClassifier(base.BaseEstimator, base.ClassifierMixin):

    @abc.abstractmethod
    def fit_one(x, y):
        """Fits """
        pass

    def fit(self, X, y=None):
        X = utils.check_array(X)
        for x, yi in STREAM_METHODS[type(X)](X, y):
            self.fit_one(x, yi)
        return self

    @abc.abstractmethod
    def predict_proba_one(x: dict) -> float:
        pass

    @abc.abstractmethod
    def predict_one(x: dict) -> bool:
        pass

    def predict_proba(self, X):
        X = utils.check_array(X)
        y_pred = np.empty(len(X))
        for i, (x, _) in enumerate(stream.iter_numpy(X, shuffle=False)):
            y_pred[i] = self.predict_proba_one(x)
        return y_pred

    def predict(self, X):
        X = utils.check_array(X)
        return np.round(self.predict_proba(X)).astype(bool)


class MultiClassifier(base.BaseEstimator, base.ClassifierMixin):

    @abc.abstractmethod
    def fit_one(x, y):
        """Fits """
        pass

    def fit(self, X, y=None):
        X = utils.check_array(X)
        for x, yi in STREAM_METHODS[type(X)](X, y):
            self.fit_one(x, yi)
        return self

    @abc.abstractmethod
    def predict_proba_one(x: dict) -> dict:
        pass

    @abc.abstractmethod
    def predict_one(x: dict) -> str:
        pass

    def predict_proba(self, X):
        X = utils.check_array(X)
        y_pred = [None] * len(X)
        for i, (x, y) in enumerate(stream.iter_numpy(X, shuffle=False)):
            y_pred[i] = list(self.fit_one(x, y).values())
        return np.array(y_pred)

    def predict(self, X):
        X = utils.check_array(X)
        y_pred = [None] * len(X)
        for i, (x, _) in enumerate(stream.iter_numpy(X, shuffle=False)):
            y_pred[i] = self.predict_one(x)
        return np.array(y_pred)


class Transformer(base.BaseEstimator, base.TransformerMixin):

    @abc.abstractmethod
    def fit_one(x, y):
        """Fits """
        pass

    def fit(self, X, y=None):
        X = utils.check_array(X)
        for x, yi in STREAM_METHODS[type(X)](X, y):
            self.fit_one(x, yi)
        return self

    @abc.abstractmethod
    def transform_one(x: dict) -> dict:
        pass

    def transform(self, X):
        X = utils.check_array(X)
        y_trans = [None] * len(X)
        for i, (x, _) in enumerate(stream.iter_numpy(X)):
            y_trans[i] = list(self.transform_one(x).values())
        return np.array(y_trans)


class Clusterer(base.BaseEstimator, base.ClusterMixin):

    @abc.abstractmethod
    def fit_one(x, y):
        """Fits """
        pass

    def fit(self, X, y=None):
        X = utils.check_array(X)
        for x, yi in STREAM_METHODS[type(X)](X, y):
            self.fit_one(x, yi)
        return self

    @abc.abstractmethod
    def predict_one(x: dict) -> dict:
        pass

    def predict(self, X):
        X = utils.check_array(X)
        y_pred = np.empty(len(X), dtype='int32')
        for i, (x, _) in enumerate(stream.iter_numpy(X, shuffle=False)):
            y_pred[i] = self.predict_one(x)
        return y_pred
