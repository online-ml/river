from abc import ABCMeta, abstractmethod
from skmultiflow.core.base import BaseEstimator


class BaseListener(BaseEstimator, metaclass=ABCMeta):
    """ An abstract class that defines the constraints for all the listener
    type objects in this framework.
    
    This class should not be instantiated, as its functions are not 
    implemented.
    
    Raises
    ------
    NotImplementedError: This is an abstract class.
    
    """
    _estimator_type = 'listener'

    def __init__(self):
        super().__init__()
        pass

    @abstractmethod
    def on_new_train_step(self, sample_id, data_buffer):
        """ At each relevant update (usually at each n_wait samples) this function
        should be called to enable the plot update.
        
        Parameters
        ----------
        sample_id: int
            The current sample id.

        data_buffer: EvaluationDataBuffer
            A buffer containing evaluation data for a single training / visualization step.
         
        """
        raise NotImplementedError
