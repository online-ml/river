__author__ = 'Guilherme Matsumoto'

from abc import ABCMeta, abstractmethod
from skmultiflow.core.base_object import BaseObject


class BaseListener(BaseObject, metaclass=ABCMeta):
    """ BaseListener
    
    An abstract class that defines the constraints for all the listener 
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
    def on_new_train_step(self, performance_point, train_step):
        """ on new_train_step
        
        At each relevant update (usually at each n_wait samples) this function 
        should be called to enable the plot update.
        
        Parameters
        ----------
        performance_point: int
            The number of samples analyzed.
        
        train_step: dictionary
            A dictionary containing all subplots to update as keys and as values 
            their new values.
         
        """
        raise NotImplementedError

    def get_class_type(self):
        return 'listener'

    @abstractmethod
    def get_info(self):
        raise NotImplementedError