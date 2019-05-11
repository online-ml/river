import copy as cp
import numpy as np
from sklearn.tree import DecisionTreeClassifier

from skmultiflow.core import BaseSKMObject, ClassifierMixin, MetaEstimatorMixin


class LearnNSE(BaseSKMObject, ClassifierMixin, MetaEstimatorMixin):
    """ Learn++.NSE ensemble classifier.

    Learn++.NSE [1]_ is an ensemble of classifiers for incremental learning
    from non-stationary environments (NSEs) where the underlying data
    distributions change over time. It learns from consecutive batches of data
    that experience constant or variable rate of drift, addition or deletion
    of concept classes, as well as cyclical drift.

    References
    ----------
    .. [1] Ryan Elwell and Robi Polikar. Incremental learning of concept drift in
       non-stationary environments. IEEE Transactions on Neural Networks,
       22(10):1517-1531, October 2011. ISSN 1045-9227. URL
       http://dx.doi.org/10.1109/TNN.2011.2160459

    Parameters
    ----------
    base_estimator: StreamModel or sklearn.BaseEstimator (default=DecisionTreeClassifier)
        Each member of the ensemble is an instance of the base estimator.
    n_estimators: int (default=15)
        The number of base estimators in the ensemble.
    window_size: int (default=250)
        The size of the training window (batch), in other words, how many instances are kept for training.
    crossing_point: float (default=0.5)
        Halfway crossing point of the sigmoid function controlling the number of previous
        periods taken into account during weighting.
    slope: float (default=0.5)
        Slope of the sigmoid function controlling the number
        of previous periods taken into account during weighting.
    pruning: string (default=None)
        Classifiers pruning strategy to be used.
        pruning=None: Don't prune classifiers
        pruning='age': Age-based
        pruning='error': Error-based
    """

    def __init__(self,
                 base_estimator=DecisionTreeClassifier(),
                 window_size=250,
                 slope=0.5,
                 crossing_point=10,
                 n_estimators=15,
                 pruning=None):
        super().__init__()
        self.ensemble = []
        self.ensemble_weights = []
        self.bkts = []
        self.wkts = []
        self.buffer = []
        self.window_size = window_size
        self.slope = slope
        self.crossing_point = crossing_point
        self.n_estimators = n_estimators
        self.pruning = pruning
        self.X_batch = []
        self.y_batch = []
        self.instance_weights = []
        self.base_estimator = cp.deepcopy(base_estimator)
        self.classes = None

    @staticmethod
    def _train_model(estimator, X, y, classes=None):
        try:
            estimator.fit(X, y)
        except (NotImplementedError, TypeError):
            estimator.partial_fit(X, y, classes=classes)

    def partial_fit(self, X, y=None, classes=None, sample_weight=None):
        """
        Partially fits the model, based on the X and y matrix.

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            Features matrix used for partially updating the model.

        y: Array-like
            An array-like of all the class labels for the samples in X.

        classes: numpy.ndarray, optional (default=None)
            Array with all possible/known class labels. This is an optional parameter, except
            for the first partial_fit call where it is compulsory.

        sample_weight: NOT used (default=None)

        Raises
        ------
        RuntimeError:
            A RuntimeError is raised if the 'classes' parameter is not
            passed in the first partial_fit call, or if they are passed in further
            calls but differ from the initial classes list passed.
            A RuntimeError is raised if the base_estimator is too weak. In other word,
            it has too low accuracy on the dataset.

        Returns
        -------
        LearnNSE
            self
        """

        N, _ = X.shape
        if self.classes is None:
            if classes is None:
                raise RuntimeError("Should pass the classes in the first partial_fit call")
            else:
                self.classes = classes

        for i in range(N):
            self.X_batch.append(X[i])
            self.y_batch.append(y[i])
            mt = len(self.y_batch)

            if mt == self.window_size:
                self.X_batch = np.array(self.X_batch)
                self.y_batch = np.array(self.y_batch)

                classifier = cp.deepcopy(self.base_estimator)

                if len(self.ensemble) > 0:
                    # Compute the error of the existing ensemble on the new data
                    votes = self.predict(self.X_batch)

                    et = np.sum(votes != self.y_batch) / mt

                    # Update and normalize instance weights
                    self.instance_weights = np.ones(mt) / mt
                    self.instance_weights[votes == self.y_batch] = et / mt

                    # normalize instance weights (distribution)
                    self.instance_weights = self.instance_weights / np.sum(self.instance_weights)

                    # Train base classifier with Dt
                    self._train_model(classifier, self.X_batch, self.y_batch, classes=self.classes)

                else:
                    # First run! train the classifier on the instances with the same weight
                    self.instance_weights = np.ones(mt) / mt

                    self._train_model(classifier, self.X_batch, self.y_batch, classes=self.classes)

                self.ensemble.append(classifier)
                self.bkts.append([])
                self.wkts.append([])
                self.ensemble_weights = []

                t = len(self.ensemble)
                max_error = -np.inf
                error_index = -1

                # Evaluate all existing classifiers on the new dataset
                for k in range(1, t + 1):
                    pred = self.ensemble[k - 1].predict(self.X_batch)
                    ekt = np.sum(self.instance_weights[pred != self.y_batch])

                    if k == t and ekt > 0.5:
                        # Generate a new classifier
                        classifier = cp.deepcopy(self.base_estimator)
                        self._train_model(classifier, self.X_batch, self.y_batch, classes=self.classes)
                        self.ensemble[k - 1] = classifier
                    elif ekt > 0.5:
                        ekt = 0.5

                    # Storing the index of the classifier with higher error in case of error-based pruning
                    if ekt > max_error:
                        max_error = ekt
                        error_index = k

                    # Normalize errors
                    bkt = ekt / (1 - ekt)

                    # store normalized error for this classifier
                    nbkts = self.bkts[k - 1]
                    nbkts.append(bkt)

                    # compute the weighted normalized errors for kth classifier h_k
                    wkt = 1.0 / (1.0 + np.exp(-self.slope * (t - k - self.crossing_point)))
                    weights = self.wkts[k - 1]
                    weights.append(wkt / (np.sum(weights) + wkt))

                    sbkt = np.sum(np.array(nbkts) * np.array(weights)) + 1e-50

                    # Calculate classifier voting weights
                    self.ensemble_weights.append(np.log(1.0 / sbkt))

                # Ensemble pruning

                if self.pruning == 'age' and t > self.n_estimators:
                    # Age-based
                    self.ensemble.pop(0)
                    self.ensemble_weights.pop(0)
                    self.bkts.pop(0)
                    self.wkts.pop(0)
                elif self.pruning == 'error' and t > self.n_estimators:
                    # Error-based
                    self.ensemble.pop(error_index - 1)
                    self.ensemble_weights.pop(error_index - 1)
                    self.bkts.pop(error_index - 1)
                    self.wkts.pop(error_index - 1)

                # Reset the buffer
                self.X_batch = []
                self.y_batch = []
        return self

    def __vote_proba(self, X, t, classes):
        res = []
        for m in range(len(X)):
            votes = np.zeros((1, len(classes)))
            for i in range(t):
                if self.ensemble_weights[i] > 0:
                    h = self.ensemble[i]
                    y_predicts = h.predict_proba(X[m].reshape(1, -1))
                    y_predicts /= np.linalg.norm(y_predicts, ord=1, axis=1, keepdims=True)
                    votes += self.ensemble_weights[i] * y_predicts

            res.append(votes.reshape(len(classes)))
        return np.array(res)

    def predict_proba(self, X):
        """ Predicts the probability of each sample belonging to each one of the
        known classes.

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            A matrix of the samples we want to predict.

        Returns
        -------
        numpy.ndarray
            An array of shape (n_samples, n_features), in which each outer entry is
            associated with the X entry of the same index. And where the list in
            index [i] contains len(self.target_values) elements, each of which represents
            the probability that the i-th sample of X belongs to a certain label.
        """

        if not self.ensemble:
            return np.zeros((len(X), 1))

        return self.__vote_proba(X, len(self.ensemble), self.classes)

    def predict(self, X):
        """ Predicts the class for a given sample by majority vote from all
        the members of the ensemble.

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            A matrix of the samples we want to predict.
        Returns
        -------
        numpy.ndarray
            A numpy.ndarray with the label prediction for all the samples in X.
        """

        votes = self.predict_proba(X)
        return np.argmax(votes, axis=1)

    def reset(self):
        self.ensemble = []
        self.ensemble_weights = []
        self.bkts = []
        self.wkts = []
        self.X_batch = []
        self.y_batch = []
