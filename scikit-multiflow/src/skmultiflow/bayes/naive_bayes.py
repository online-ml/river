import numpy as np

from collections import deque

from skmultiflow.core import BaseSKMObject, ClassifierMixin
from skmultiflow.utils import get_dimensions
from skmultiflow.bayes import do_naive_bayes_prediction
from skmultiflow.trees._attribute_observer import NumericAttributeClassObserverGaussian
from skmultiflow.trees._attribute_observer import NominalAttributeClassObserver


class NaiveBayes(BaseSKMObject, ClassifierMixin):
    """ Naive Bayes classifier.

    Performs classic bayesian prediction while making naive assumption that all inputs are
    independent. Naive Bayes is a classifier algorithm known for its simplicity
    and low computational cost. Given `n` different classes, the trained Naive Bayes classifier
    predicts for every unlabelled instance the class to which it belongs with high accuracy.

    Parameters
    ----------
    nominal_attributes: numpy.ndarray (optional, default=None)
        List of Nominal attributes. If emtpy, then assume that all attributes are numerical.

    Notes
    -----
    The `scikit-learn` implementations of NaiveBayes are compatible with `scikit-multiflow`
    with the caveat that they must be partially fitted before use. In the `scikit-multiflow`
    evaluators this is done by setting `pretrain_size>0`.

    Examples
    --------
    >>> # Imports
    >>> from skmultiflow.data import SEAGenerator
    >>> from skmultiflow.bayes import NaiveBayes
    >>>
    >>> # Setup a data stream
    >>> stream = SEAGenerator(random_state=1)
    >>>
    >>> # Setup Naive Bayes estimator
    >>> naive_bayes = NaiveBayes()
    >>>
    >>> # Setup variables to control loop and track performance
    >>> n_samples = 0
    >>> correct_cnt = 0
    >>> max_samples = 200
    >>>
    >>> # Train the estimator with the samples provided by the data stream
    >>> while n_samples < max_samples and stream.has_more_samples():
    >>>     X, y = stream.next_sample()
    >>>     y_pred = naive_bayes.predict(X)
    >>>     if y[0] == y_pred[0]:
    >>>         correct_cnt += 1
    >>>     naive_bayes.partial_fit(X, y)
    >>>     n_samples += 1
    >>>
    >>> # Display results
    >>> print('{} samples analyzed.'.format(n_samples))
    >>> print('Naive Bayes accuracy: {}'.format(correct_cnt / n_samples))

    """

    def __init__(self, nominal_attributes=None):
        super().__init__()
        self._observed_class_distribution = {}
        self._attribute_observers = {}
        self._classes = None
        self.nominal_attributes = nominal_attributes
        if not self.nominal_attributes:
            self._nominal_attributes = []
        else:
            self._nominal_attributes = self.nominal_attributes

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        """ Partially (incrementally) fit the model.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The features to train the model.

        y: numpy.ndarray of shape (n_samples)
            An array-like with the labels of all samples in X.

        classes: numpy.ndarray, optional (default=None)
            Array with all possible/known classes. Usage varies depending on the learning method.

        sample_weight: numpy.ndarray of shape (n_samples), optional (default=None)
            Samples weight. If not provided, uniform weights are assumed.
            Usage varies depending on the learning method.

        Returns
        -------
        NaiveBayes
            self

        """
        if not self._classes and classes is not None:
            self._classes = classes

        if y is not None:
            row_cnt, _ = get_dimensions(X)
            if sample_weight is None:
                sample_weight = np.ones(row_cnt)
            if row_cnt != len(sample_weight):
                raise ValueError(
                    'Inconsistent number of instances ({}) and weights ({}).'.format(row_cnt, len(
                        sample_weight)))
            for i in range(row_cnt):
                if sample_weight[i] != 0.0:
                    self._partial_fit(X[i], y[i], sample_weight[i])
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
            obs.update(X[i], int(y), weight)

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
        r, _ = get_dimensions(X)
        predictions = deque()
        y_proba = self.predict_proba(X)
        for i in range(r):
            class_val = np.argmax(y_proba[i])
            predictions.append(class_val)
        return np.array(predictions)

    def predict_proba(self, X):
        """ Estimates the probability of each sample in X belonging to each of the class-labels.

        Parameters
        ----------
        X : Numpy.ndarray of shape (n_samples, n_features)
            The matrix of samples one wants to predict the class probabilities for.

        Returns
        -------
        A numpy.ndarray of shape (n_samples, n_labels), in which each outer entry is associated
        with the X entry of the same index. And where the list in index [i] contains
        len(self.target_values) elements, each of which represents the probability that
        the i-th sample of X belongs to a certain class-label.

        """
        predictions = deque()
        r, _ = get_dimensions(X)
        if self._observed_class_distribution == {}:
            # Model is empty, all classes equal, default to zero
            return np.zeros((r, 1))
        else:
            for i in range(r):
                votes = do_naive_bayes_prediction(X[i], self._observed_class_distribution,
                                                  self._attribute_observers)
                sum_values = sum(votes.values())
                if self._classes is not None:
                    y_proba = np.zeros(int(max(self._classes)) + 1)
                else:
                    y_proba = np.zeros(int(max(votes.keys())) + 1)
                for key, value in votes.items():
                    y_proba[int(key)] = value / sum_values if sum_values != 0 else 0.0
                predictions.append(y_proba)
        return np.array(predictions)
