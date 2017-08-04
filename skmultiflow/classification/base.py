__author__ = 'Jacob Montiel'

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
        """ Initialization.
        """
        pass

    @abstractmethod
    def fit(self, X, y, classes = None):
        """ fit
        
        Fit the model under a batch setting.

        Parameters
        ----------
        X : Numpy.ndarray of shape (n_samples, n_features)
            The array of samples used to fit the model.
            
        y: Array-like
            An array-like with the labels of all samples in X.
            
        classes: Array-like
            Optional parameter that contains all labels that may appear 
            in samples.
            
        Returns
        -------
        self
            
        """
        raise NotImplementedError

    @abstractmethod
    def partial_fit(self, X, y, classes=None):
        """ partial_fit
        
        Partial (incremental) fit the model under an online learning setting.

        Parameters
        ----------
        X : Numpy.ndarray of shape (n_samples, n_features)
            The array of samples used to fit the model.
            
        y: Array-like
            An array-like with the labels of all samples in X.
            
        classes: Array-like
            Contains all labels that may appear in samples. It's an optional 
            parameter, except during the first partial_fit call, when it's 
            obligatory.
        
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
        raise NotImplementedError

    @abstractmethod
    def score(self, X, y):
        raise NotImplementedError

    def get_class_type(self):
        return 'estimator'

    @abstractmethod
    def get_info(self):
        raise NotImplementedError