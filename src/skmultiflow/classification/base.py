from abc import ABCMeta, abstractmethod
from skmultiflow.core.base_object import BaseObject


class BaseClassifier(BaseObject, metaclass=ABCMeta):
    """ BaseClassifier
    
    The abstract class that works as a base model for all of this framework's 
    classifiers. It creates a basic interface that classifiers should follow 
    in order to use them with all the tools available in scikit-workflow.
    
    This class should not me instantiated, as none of its methods, except the 
    get_class_type, are implemented.
    
    Raises
    ------
    NotImplementedError: This is an abstract class.
    
    """

    def __init__(self):
        super().__init__()


    @abstractmethod
    def fit(self, X, y, classes=None, weight=None):
        """ fit
        
        Fit the model under a batch setting.

        Parameters
        ----------
        X : Numpy.ndarray of shape (n_samples, n_features)
            The array of samples used to fit the model.
            
        y: Array-like
            An array-like with the labels of all samples in X.
            
        classes: Array-like, optional
            Contains all labels that may appear in samples. Applicability varies depending on the algorithm.

        weight: Array-like, optional
            Instance weight. If not provided, uniform weights are assumed.
            Applicability varies depending on the algorithm.
            
        Returns
        -------
        self
            
        """
        raise NotImplementedError

    @abstractmethod
    def partial_fit(self, X, y, classes=None, weight=None):
        """ partial_fit
        
        Partial (incremental) fit the model under an online learning setting.

        Parameters
        ----------
        X : Numpy.ndarray of shape (n_samples, n_features)
            The array of samples used to fit the model.
            
        y: Array-like
            An array-like with the labels of all samples in X.
            
        classes: Array-like, optional
            Contains all labels that may appear in samples. Applicability varies depending on the algorithm.

        weight: Array-like, optional
            Instance weight. If not provided, uniform weights are assumed.
            Applicability varies depending on the algorithm.
        
        Returns
        -------
        self
        
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, X):
        """ predict
        
        Predicts labels using the model.

        Parameters
        ----------
        X : Numpy.ndarray of shape (n_samples, n_features)
            The matrix of samples one wants to predict.

        Returns
        -------
        An array-like with all the predictions for the samples in X.
        
        """
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, X):
        """ predict_proba
        
        Estimates the probability of each sample in X belonging to each of 
        the existing labels.

        Parameters
        ----------
        X : Numpy.ndarray of shape (n_samples, n_features)
            The matrix of samples one wants to predict.

        Returns
        -------
        An array of shape (n_samples, n_labels), in which each outer entry is 
        associated with the X entry of the same index. And where the list in 
        index [i] contains len(self.classes) elements, each of which represents 
        the probability that the i-th sample of X belongs to a certain label.
        
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        """ reset
        
        Resets the classifier's parameters.
        
        Returns
        -------
        BaseClassifier
            self
        
        """
        raise NotImplementedError

    @abstractmethod
    def score(self, X, y):
        """ score
        
        This function isn't natively implemented on all classifiers. But if 
        it's, it will return the performance base metric for that classifier 
        at that moment and having the X matrix as input.
        
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
        raise NotImplementedError

    def get_class_type(self):
        return 'estimator'

    @abstractmethod
    def get_info(self):
        raise NotImplementedError
