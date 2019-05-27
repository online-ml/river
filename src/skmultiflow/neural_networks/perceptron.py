import numpy as np
from skmultiflow.core import BaseSKMObject, ClassifierMixin
from sklearn.linear_model import Perceptron


class PerceptronMask(BaseSKMObject, ClassifierMixin):
    """ Mask for sklearn.linear_model.Perceptron.

    scikit-multiflow requires a few interfaces, not present in scikit-learn,
    This mask serves as a wrapper for the Perceptron classifier.

    """
    def __init__(self,
                 penalty=None,
                 alpha=0.0001,
                 fit_intercept=True,
                 max_iter=None,
                 tol=None,
                 shuffle=True,
                 verbose=0,
                 eta0=1.0,
                 n_jobs=None,
                 random_state=0,
                 early_stopping=False,
                 validation_fraction=0.1,
                 n_iter_no_change=5,
                 class_weight=None,
                 warm_start=False,
                 n_iter=None):
        self.penalty = penalty
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.shuffle = shuffle
        self.verbose = verbose
        self.eta0 = eta0
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.class_weight = class_weight
        self.warm_start = warm_start
        self.n_iter = n_iter
        super().__init__()
        self.classifier = Perceptron(penalty=self.penalty,
                                     alpha=self.alpha,
                                     fit_intercept=self.fit_intercept,
                                     max_iter=self.max_iter,
                                     tol=self.tol,
                                     shuffle=self.shuffle,
                                     verbose=self.verbose,
                                     eta0=self.eta0,
                                     n_jobs=self.n_jobs,
                                     random_state=self.random_state,
                                     early_stopping=self.early_stopping,
                                     validation_fraction=self.validation_fraction,
                                     n_iter_no_change=self.n_iter_no_change,
                                     class_weight=self.class_weight,
                                     warm_start=self.warm_start)

    def fit(self, X, y, classes=None, sample_weight=None):
        """ Calls the Perceptron fit function from sklearn.

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            The feature's matrix.

        y: Array-like
            The class labels for all samples in X.

        classes: Not used.

        sample_weight:
            Samples weight. If not provided, uniform weights are assumed.

        Returns
        -------
        PerceptronMask
            self

        """
        self.classifier.fit(X=X, y=y, sample_weight=sample_weight)
        return self

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        """ partial_fit

        Calls the Perceptron partial_fit from sklearn.

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            The feature's matrix.

        y: Array-like
            The class labels for all samples in X.

        classes: Not used.

        sample_weight:
            Samples weight. If not provided, uniform weights are assumed.

        Returns
        -------
        PerceptronMask
            self

        """
        self.classifier.partial_fit(X=X, y=y, classes=classes, sample_weight=sample_weight)
        return self

    def predict(self, X):
        """ predict

        Uses the current model to predict samples in X.

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            The feature's matrix.

        Returns
        -------
        numpy.ndarray
            A numpy.ndarray containing the predicted labels for all instances in X.

        """
        return np.asarray(self.classifier.predict(X))

    def predict_proba(self, X):
        """ Predicts the probability of each sample belonging to each one of the known classes.
    
        Parameters
        ----------
        X: Numpy.ndarray of shape (n_samples, n_features)
            A matrix of the samples we want to predict.
    
        Returns
        -------
        numpy.ndarray
            An array of shape (n_samples, n_features), in which each outer entry is 
            associated with the X entry of the same index. And where the list in 
            index [i] contains len(self.target_values) elements, each of which represents
            the probability that the i-th sample of X belongs to a certain label.
    
        """
        return self.classifier._predict_proba_lr(X)
