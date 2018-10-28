from abc import ABCMeta, abstractmethod
from skmultiflow.core.base_object import BaseObject


class StreamModel(BaseObject, metaclass=ABCMeta):
    """
    The abstract class for stream models. It provides the template that models must follow in scikit-multiflow.

    Notes
    _____
    This class should not me instantiated, as none of its methods, except the `get_class_type`, are implemented.

    Raises
    ------
    NotImplementedError: This is an abstract class.
    
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def fit(self, X, y, classes=None, weight=None):
        """ Fit the model under the batch setting.

        Parameters
        ----------
        X : Numpy.ndarray of shape (n_samples, n_features)
            The array of samples used to fit the model.

        y: Array-like
            An array-like with the labels of all samples in X.

        classes: Array-like, optional (default=None)
            Contains all possible labels. Applicability varies depending on the algorithm.

        weight: Array-like, optional (default=None)
            Instance weight. If not provided, uniform weights are assumed.
            Applicability varies depending on the algorithm.

        Returns
        -------
        self

        """
        raise NotImplementedError

    @abstractmethod
    def partial_fit(self, X, y, classes=None, weight=None):
        """ Partial (incremental) fit the model under the stream learning setting.

        Parameters
        ----------
        X : Numpy.ndarray of shape (n_samples, n_features)
            The array of samples used to fit the model.

        y: Array-like
            An array-like with the labels of all samples in X.

        classes: Array-like, optional (default=None)
            Contains all possible labels. Applicability varies depending on the algorithm.

        weight: Array-like, optional (default=None)
            Instance weight. If not provided, uniform weights are assumed.
            Applicability varies depending on the algorithm.

        Returns
        -------
        self

        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, X):
        """ Predicts targets using the model.

        Parameters
        ----------
        X : Numpy.ndarray of shape (n_samples, n_features)
            The matrix of samples one wants to predict the labels for.

        Returns
        -------
        An array-like with all the predictions for the samples in X.

        """
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, X):
        """ On classifiers, estimates the probability of each sample in X belonging to each of the existing labels.

        Parameters
        ----------
        X : Numpy.ndarray of shape (n_samples, n_features)
            The matrix of samples one wants to predict the class probabilities for.

        Returns
        -------
        An array of shape (n_samples, n_labels), in which each outer entry is associated with the X entry of the same
        index. And where the list in index [i] contains len(self.target_values) elements, each of which represents the
        probability that the i-th sample of X belongs to a certain label.

        """
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        """ Resets the model to its initial state.

        Returns
        -------
        StreamModel
            self

        """
        raise NotImplementedError

    @abstractmethod
    def score(self, X, y):
        """ score

        Calculate the performance base metric for the model in its current state.

        Notes
        -----
        This function isn't natively implemented on all models.

        Parameters
        ----------
        X: numpy.ndarray of shape (n_sample, n_features)
            The features matrix.

        y: Array-like
            An array-like containing the targets for all samples in X.

        Returns
        -------
        float
            The model's score.

        """
        raise NotImplementedError

    def get_class_type(self):
        return 'estimator'

    @abstractmethod
    def get_info(self):
        raise NotImplementedError
