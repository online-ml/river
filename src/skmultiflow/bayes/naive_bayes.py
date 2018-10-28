from skmultiflow.core.base import StreamModel
from sklearn.naive_bayes import MultinomialNB


class NaiveBayes(StreamModel):
    """ NaiveBayes
    
    A mask for scikit-learn's Naive Bayes classifier.
    
    Because scikit-multiflow's framework requires a few interfaces, not present
    in scikit-learn, this mask allows the first to use classifiers native to
    the latter.
    
    """
    def __init__(self):
        super().__init__()
        self.classifier = MultinomialNB()

    def fit(self, X, y, classes=None, weight=None):
        """ fit
        
        Calls the MultinomialNB fit function from sklearn.
        
        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            The feature's matrix.
        
        y: Array-like
            The class labels for all samples in X.
            
        classes: list, optional
            A list with all the possible labels of the classification problem.

        weight: Array-like.
            Instance weight. If not provided, uniform weights are assumed.

        Returns
        -------
        NaiveBayes
            self
        
        """
        self.classifier.fit(X, y, sample_weight=weight)
        return self

    def partial_fit(self, X, y, classes=None, weight=None):
        """ partial_fit

        Calls the MultinomialNB partial_fit from sklearn.

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            The feature's matrix.

        y: Array-like
            The class labels for all samples in X.

        classes: list, optional
            A list with all the possible labels of the classification problem.

        weight: Not used.

        Returns
        -------
        NaiveBayes
            self

        """
        self.classifier.partial_fit(X, y, classes)
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
        if not hasattr(self.classifier, "classes_"):
            return [0]
        return self.classifier.predict(X)

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
        if not hasattr(self.classifier, "classes_"):
            return [0.0]
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

    def reset(self):
        self.__init__()

    def get_info(self):
        return 'Multinomial Naive Bayes classifier'
