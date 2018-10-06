from abc import ABCMeta, abstractmethod
from skmultiflow.core.base_object import BaseObject


class BaseListener(BaseObject, metaclass=ABCMeta):
    """ An abstract class that defines the constraints for all the listener
    type objects in this framework.
    
    This class should not be instantiated, as its functions are not 
    implemented.
    
    Raises
    ------
    NotImplementedError: This is an abstract class.
    
    """

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

    def get_class_type(self):
        return 'listener'

    @abstractmethod
    def get_info(self):
        raise NotImplementedError
