__author__ = 'Guilherme Matsumoto'

from abc import ABCMeta, abstractmethod
from skmultiflow.core.base_object import BaseObject


class BaseEvaluator(BaseObject, metaclass=ABCMeta):
    """ BaseEvaluator

    The abstract class that works as a base model for all of this framework's 
    evaluators. It creates a basic interface that evaluation modules should 
    follow in order to use them with all the tools available in scikit-workflow.

    This class should not me instantiated, as none of its methods, except the 
    get_class_type, are implemented.

    Raises
    ------
    NotImplementedError: This is an abstract class.

    """

    def __init__(self):
        super().__init__()
        pass

    @abstractmethod
    def eval(self, stream, classifier):
        """ eval
        
        This function evaluates the classifier, using the class parameters, and 
        by feeding it with instances coming from the stream parameter.
        
        Parameters
        ----------
        stream: BaseInstanceStream extension
            The stream to be use in the evaluation process.
        
        classifier: BaseClassifier extension or list of BaseClassifier extensions
            The classifier or classifiers to be evaluated.
            
        Returns
        -------
        BaseClassifier extension or list of BaseClassifier extensions
            The trained classifier's at the end of the evaluation process.
            
        """
        raise NotImplementedError

    @abstractmethod
    def partial_fit(self, X, y, classes=None, weight=None):
        """ partial_fit
        
        Partially fits the classifiers.
        
        X: numpy.ndarray of shape (n_samples, n_features)
            The feature's matrix.
        
        y: Array-like
            An array-like containing the class labels of all samples in X.
        
        classes: list
            A list containing all class labels of the classification problem.

        weight: Array-like
            Instance weight. If not provided, uniform weights are assumed.
            Applicability varies depending on the algorithm.

        Returns
        -------
        BaseClassifier extension or list of BaseClassifier extensions
            The trained classifier's at the end of the evaluation process.
        
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, X):
        """ predict
        
        Predicts with the classifier, or classifiers, being evaluated.
        
        X: numpy.ndarray of shape (n_samples, n_features)
            The feature's matrix.
        
        Returns
        -------
        list
            A list containing the array-likes representing each classifier's 
            prediction.
        
        """
        raise NotImplementedError

    def get_class_type(self):
        return 'evaluator'

    @abstractmethod
    def set_params(self, dict):
        """ set_params
        
        Pass parameter names and values through a dictionary so that their 
        values can be updated.
        
        Parameters
        ----------
        dict: dictionary
            A dictionary where the keys are parameters' names and the values 
            are the new values for those parameters.
         
        """
        raise NotImplementedError

    @abstractmethod
    def _update_metrics(self):
        """ _update_metrics
        
        Updates the classifiers' evaluation metrics.
        
        """
        raise NotImplementedError

    @abstractmethod
    def get_info(self):
        raise NotImplementedError
