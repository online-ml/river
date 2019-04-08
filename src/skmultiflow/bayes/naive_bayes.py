from collections import deque
import numpy as np
from skmultiflow.core.base import StreamModel
from skmultiflow.utils import get_dimensions
from skmultiflow.trees.numeric_attribute_class_observer_gaussian import NumericAttributeClassObserverGaussian
from skmultiflow.trees.nominal_attribute_class_observer import NominalAttributeClassObserver
from skmultiflow.bayes import do_naive_bayes_prediction


class NaiveBayes(StreamModel):
    """ Performs classic bayesian prediction while making naive assumption that all inputs are independent.
    Naive Bayes is a classifier algorithm known for its simplicity and low computational cost. Given `n` different
    classes, the trained Naive Bayes classifier predicts for every unlabelled instance the class to which it
    belongs with high accuracy.

    Parameters
    ----------
    nominal_attributes: array-like (optional)
        List of Nominal attributes. If emtpy, then assume that all attributes are numerical.

    Notes
    -----
    The `scikit-learn` implementations of NaiveBayes are compatible with `scikit-multiflow` with the caveat that
    they must be partially fitted before use. In the `scikit-multiflow` evaluators this is done by setting
    `pretrain_size>0`.

    """

    def __init__(self, nominal_attributes=None):
        super().__init__()
        self._observed_class_distribution = {}
        self._attribute_observers = {}
        self._classes = None
        if not nominal_attributes:
            self._nominal_attributes = []
        else:
            self._nominal_attributes = nominal_attributes

    def fit(self, X, y, classes=None, weight=None):
        """ Fits the model.

        Parameters
        ----------
        X: numpy.ndarray, shape (n_samples, n_features)
            The feature's matrix.

        y: numpy.ndarray, shape (n_samples)
            The class labels for all samples in X.

        classes: numpy.ndarray, shape (n_samples), optional (default=None)
            A list with all the possible labels of the classification problem.

        weight: numpy.ndarray, shape (n_samples), optional (default=None)
            Instance(s) weight(s). If not provided, uniform weights are assumed.

        Returns
        -------
        NaiveBayes
            self

        """
        self.partial_fit(X, y, classes=None, weight=None)

    def partial_fit(self, X, y, classes=None, weight=None):
        """ Partially fits the model.

        Parameters
        ----------
        X: numpy.ndarray, shape (n_samples, n_features)
            The feature's matrix.

        y: numpy.ndarray, shape (n_samples)
            The class labels for all samples in X.

        classes: numpy.ndarray, shape (n_samples), optional (default=None)
            A list with all the possible labels of the classification problem.

        weight: numpy.ndarray, shape (n_samples), optional (default=None)
            Instance(s) weight(s). If not provided, uniform weights are assumed.

        Returns
        -------
        NaiveBayes
            self

        """
        if not self._classes and classes is not None:
            self._classes = classes

        if y is not None:
            row_cnt, _ = get_dimensions(X)
            if weight is None:
                weight = np.ones(row_cnt)
            if row_cnt != len(weight):
                raise ValueError('Inconsistent number of instances ({}) and weights ({}).'.format(row_cnt, len(weight)))
            for i in range(row_cnt):
                if weight[i] != 0.0:
                    self._partial_fit(X[i], y[i], weight[i])
        return self

    def _partial_fit(self, X, y, weight):
        try:
            self._observed_class_distribution[y] += weight
        except KeyError:
            self._observed_class_distribution[y] = weight
        for i in range(len(X)):
            try:
                obs = self._attribute_observers[i]
            except KeyError:
                if i in self._nominal_attributes:
                    obs = NominalAttributeClassObserver()
                else:
                    obs = NumericAttributeClassObserverGaussian()
                self._attribute_observers[i] = obs
            obs.observe_attribute_class(X[i], int(y), weight)

    def predict(self, X):
        """ Uses the current model to predict samples in X.

        Parameters
        ----------
        X: numpy.ndarray, shape (n_samples, n_features)
            The feature's matrix.

        Returns
        -------
        numpy.ndarray
            An array containing the predicted labels for all instances in X.

        """
        r, _ = get_dimensions(X)
        predictions = deque()
        y_proba = self.predict_proba(X)
        for i in range(r):
            class_val = np.argmax(y_proba[i])
            predictions.append(class_val)
        return np.array(predictions)

    def predict_proba(self, X):
        """ Predicts the probability of each sample belonging to each one of the
        known classes.

        Parameters
        ----------
        X: Numpy.ndarray, shape (n_samples, n_features)
            A matrix of the samples we want to predict.

        Returns
        -------
        numpy.ndarray
            An array of shape (n_samples, n_features), in which each outer entry is
            associated with the X entry of the same index. And where the list in
            index [i] contains len(self.target_values) elements, each of which represents
            the probability that the i-th sample of X belongs to a certain label.

        """
        predictions = deque()
        if self._observed_class_distribution == {}:
            # Model is empty, all classes equal, default to zero
            r, _ = get_dimensions(X)
            return np.zeros(r)
        else:
            r, _ = get_dimensions(X)
            for i in range(r):
                votes = do_naive_bayes_prediction(X[i], self._observed_class_distribution, self._attribute_observers)
                sum_values = sum(votes.values())
                if self._classes is not None:
                    y_proba = np.zeros(int(max(self._classes)) + 1)
                else:
                    y_proba = np.zeros(int(max(votes.keys())) + 1)
                for key, value in votes.items():
                    y_proba[int(key)] = value / sum_values if sum_values != 0 else 0.0
                predictions.append(y_proba)
        return np.array(predictions)

    def score(self, X, y):
        from sklearn.metrics import accuracy_score
        return accuracy_score(y_true=y, y_pred=self.predict(X))

    def reset(self):
        self.__init__(self._nominal_attributes)

    def get_info(self):
        """ Collects information about the classifier's configuration.

        Returns
        -------
        string
            Configuration for this classifier instance.
        """
        description = type(self).__name__ + ': '
        description += 'nominal attributes: {} - '.format(self._nominal_attributes)
        return description
