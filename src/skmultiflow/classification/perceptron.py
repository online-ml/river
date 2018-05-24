from skmultiflow.core.base import StreamModel
from sklearn.linear_model.perceptron import Perceptron


class PerceptronMask(StreamModel):
    """ PerceptronMask

    A mask for scikit-learn's Perceptron classifier.

    Because scikit-multiflow's framework require a few interfaces, not present 
    int scikit-learn, this mask allows the first to use classifiers native to 
    the latter.

    """
    def __init__(self):
        super().__init__()
        self.classifier = Perceptron(n_iter=50)

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
        list
            A list containing the predicted labels for all instances in X.

        """
        return self.classifier.predict(X)

    def predict_proba(self, X):
        """ predict_proba

        Predicts the probability of each sample belonging to each one of the 
        known target_values.
    
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
        return self.classifier.predict_proba(X)

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
        penalty = params['penalty']
        penalty = 'None' if penalty is None else penalty
        fit_int = params['fit_intercept']
        fit_int = 'True' if fit_int else 'False'
        shuffle = params['shuffle']
        shuffle = 'True' if shuffle else 'False'
        return 'Perceptron: penalty: ' + penalty + \
               '  -  alpha: ' + str(round(params['alpha'], 3)) + \
               '  -  fit_intercept: ' + fit_int + \
               '  -  n_iter: ' + str(params['n_iter']) + \
               '  -  shuffle: ' + shuffle
