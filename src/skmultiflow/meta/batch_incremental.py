import copy as cp
from inspect import signature

import numpy as np

from skmultiflow.core import BaseSKMObject, ClassifierMixin, MetaEstimatorMixin
from sklearn.tree import DecisionTreeClassifier


class BatchIncremental(BaseSKMObject, ClassifierMixin, MetaEstimatorMixin):
    """ Batch Incremental ensemble classifier.

    This is a wrapper that allows the application of any batch model to a 
    stream by incrementally building an ensemble of instances of the batch model.
    A window of examples is collected, then used to train a new model, which is
    added to the ensemble. A maximum number of models ensures memory use is finite
    (the oldest model is deleted when this number is exceeded).

    Parameters
    ----------
    base_estimator: skmultiflow.core.BaseSKMObject or sklearn.BaseEstimator (default=DecisionTreeClassifier)
        Each member of the ensemble is an instance of the base estimator

    window_size: int (default=100)
        The size of the training window (batch), in other words, how many instances are kept for training.

    n_estimators: int (default=100)
        Number of estimators in the ensemble.

    Notes
    -----
    Not yet multi-label capable.

    """

    def __init__(self, base_estimator=DecisionTreeClassifier(), window_size=100, n_estimators=100):
        self.window_size = window_size
        self.n_estimators = n_estimators
        self.base_estimator = base_estimator
        # The ensemble 
        self.ensemble = []
        self.i = -1
        self.X_batch = None
        self.y_batch = None
        self.sample_weight = None

    def partial_fit(self, X, y=None, classes=None, sample_weight=None):
        """ Partially (incrementally) fit the model.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The features to train the model.

        y: numpy.ndarray of shape (n_samples)
            An array-like with the labels of all samples in X.

        classes: Not used (default=None)

        sample_weight: numpy.ndarray of shape (n_samples), optional (default=None)
            Samples weight. If not provided, uniform weights are assumed.

        Returns
        -------
            self

        """
        N, D = X.shape

        if self.i < 0:
            # No models yet -- initialize
            self.X_batch = np.zeros((self.window_size, D))
            self.y_batch = np.zeros(self.window_size)
            self.sample_weight = np.zeros(N)
            self.i = 0

        for n in range(N):
            # For each instance ...
            # TODO not very pythonic at the moment
            self.X_batch[self.i] = X[n]
            self.y_batch[self.i] = y[n]
            self.sample_weight[self.i] = sample_weight[n] if sample_weight else 1.0

            self.i = self.i + 1
            if self.i == self.window_size:
                # Get rid of the oldest model
                if len(self.ensemble) >= self.n_estimators:
                    self.ensemble.pop(0)
                # A new model
                h = cp.deepcopy(self.base_estimator)
                # Train it
                if 'sample_weight' in signature(h.fit).parameters:
                    h.fit(X=self.X_batch, y=self.y_batch.astype(int), sample_weight=sample_weight)
                else:
                    h.fit(X=self.X_batch, y=self.y_batch.astype(int))
                # Add it
                self.ensemble.append(h)
                # Reset the window
                self.i = 0

        return self

    def predict_proba(self, X):
        """ Estimates the probability of each sample in X belonging to each of the class-labels.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The matrix of samples one wants to predict the class probabilities for.

        Returns
        -------
        A numpy.ndarray of shape (n_samples, n_labels), in which each outer entry is associated with the X entry of the
        same index. And where the list in index [i] contains len(self.target_values) elements, each of which represents
        the probability that the i-th sample of X belongs to a certain class-label.

        """
        N, D = X.shape
        votes = np.zeros(N)
        if len(self.ensemble) <= 0:
            # No models yet, just predict zeros
            return votes
        for h_i in self.ensemble:
            # Add vote (normalized by number of models)
            votes = votes + 1. / len(self.ensemble) * h_i.predict(X)
        return votes

    def predict(self, X):
        """ Predict classes for the passed data.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The set of data samples to predict the labels for.

        Returns
        -------
        A numpy.ndarray with all the predictions for the samples in X.

        """
        votes = self.predict_proba(X)
        # Suppose a threshold of 0.5
        return (votes >= 0.5) * 1.

    def reset(self):
        """ Resets the estimator to its initial state.

        Returns
        -------
            self

        """
        self.ensemble = []
        self.i = -1
        self.X_batch = None
        self.y_batch = None
        return self
