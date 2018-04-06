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
    # Constants
    PERFORMANCE = 'performance'
    KAPPA = 'kappa'
    KAPPA_T = 'kappa_t'
    KAPPA_M = 'kappa_m'
    HAMMING_SCORE = 'hamming_score'
    HAMMING_LOSS = 'hamming_loss'
    EXACT_MATCH = 'exact_match'
    J_INDEX = 'j_index'
    MSE = 'mean_square_error'
    MAE = 'mean_absolute_error'
    TRUE_VS_PREDICT = 'true_vs_predicts'
    PLOT_TYPES = [PERFORMANCE,
                  KAPPA,
                  KAPPA_T,
                  KAPPA_M,
                  HAMMING_SCORE,
                  HAMMING_LOSS,
                  EXACT_MATCH,
                  J_INDEX,
                  MSE,
                  MAE,
                  TRUE_VS_PREDICT]
    CLASSIFICATION_METRICS = [PERFORMANCE,
                              KAPPA,
                              KAPPA_T,
                              KAPPA_M,
                              TRUE_VS_PREDICT]
    REGRESSION_METRICS = [MSE,
                          MAE,
                          TRUE_VS_PREDICT]
    MULTI_OUTPUT_METRICS = [HAMMING_SCORE,
                            HAMMING_LOSS,
                            EXACT_MATCH,
                            J_INDEX]
    CLASSIFICATION = 'classification'
    REGRESSION = 'regression'
    MULTI_OUTPUT = 'multi_output'
    TASK_TYPES = [CLASSIFICATION,
                  REGRESSION,
                  MULTI_OUTPUT]

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
    def set_params(self, parameter_dict):
        """ set_params
        
        Pass parameter names and values through a dictionary so that their 
        values can be updated.
        
        Parameters
        ----------
        parameter_dict: dictionary
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
