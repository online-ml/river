import numpy as np
from skmultiflow.core.base import StreamModel
from sklearn.linear_model.perceptron import Perceptron


class PerceptronMask(StreamModel):
    """ PerceptronMask

    A mask for scikit-learn's Perceptron classifier.

    Because scikit-multiflow's framework require a few interfaces, not present 
    int scikit-learn, this mask allows the first to use classifiers native to 
    the latter.

    """
    def __init__(self, penalty=None, alpha=0.0001, fit_intercept=True,
                 max_iter=1000, tol=1e-3, shuffle=True, verbose=0, eta0=1.0,
                 n_jobs=1, random_state=0, class_weight=None,
                 warm_start=False):
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
        self.class_weight = class_weight
        self.warm_start = warm_start
        super().__init__()
        self.classifier = Perceptron(penalty=self.penalty,
                                     alpha=self.alpha,
                                     fit_intercept=self.fit_intercept,
                                     max_iter=self.max_iter,
                                     tol=self.tol,
                                     shuffle=self.shuffle,
                                     verbose=self.verbose,
                                     random_state=self.random_state,
                                     eta0=self.eta0,
                                     warm_start=self.warm_start,
                                     class_weight=self.class_weight,
                                     n_jobs=self.n_jobs)

    def fit(self, X, y, classes=None, weight=None):
        """ fit

        Calls the Perceptron fit function from sklearn.

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            The feature's matrix.

        y: Array-like
            The class labels for all samples in X.

        classes: Not used.

        weight: Instance weight. If not provided, uniform weights are assumed.

        Returns
        -------
        PerceptronMask
            self

        """
        self.classifier.fit(X, y, sample_weight=weight)
        return self

    def partial_fit(self, X, y, classes=None, weight=None):
        """ partial_fit

        Calls the Perceptron partial_fit from sklearn.

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            The feature's matrix.

        y: Array-like
            The class labels for all samples in X.

        classes: list, optional
            A list with all the possible labels of the classification problem.

        weight: Instance weight. If not provided, uniform weights are assumed.

        Returns
        -------
        PerceptronMask
            self

        """
        self.classifier.partial_fit(X, y, classes, weight)
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
        """ predict_proba

        Predicts the probability of each sample belonging to each one of the 
        known classes.
    
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

    def score(self, X, y):
        """ score

        Returns the predict performance for the samples in X.

        Parameters
        ----------
        X: numpy.ndarray of shape (n_sample, n_features)
            The features matrix.

        y: Array-like
            An array-like containing the class labels for all samples in X.

        Returns
        -------
        float
            The classifier's score.

        """
        return self.classifier.score(X, y)

    def get_info(self):
        params = self.classifier.get_params()
        info = type(self).__name__ + ':'
        info += ' - penalty: {}'.format( params['penalty'])
        info += ' - alpha: {}'.format(params['alpha'])
        info += ' - fit_intercept: {}'.format(params['fit_intercept'])
        info += ' - max_iter: {}'.format(params['max_iter'])
        info += ' - tol: {}'.format(params['tol'])
        info += ' - shuffle: {}'.format(params['shuffle'])
        info += ' - eta0: {}'.format(params['eta0'])
        info += ' - warm_start: {}'.format(params['warm_start'])
        info += ' - class_weight: {}'.format(params['class_weight'])
        info += ' - n_jobs: {}'.format(params['n_jobs'])
        return info

    def reset(self):
        self.__init__(penalty=self.penalty,
                      alpha=self.alpha,
                      fit_intercept=self.fit_intercept,
                      max_iter=self.max_iter,
                      tol=self.tol,
                      shuffle=self.shuffle,
                      verbose=self.verbose,
                      random_state=self.random_state,
                      eta0=self.eta0,
                      warm_start=self.warm_start,
                      class_weight=self.class_weight,
                      n_jobs=self.n_jobs)
