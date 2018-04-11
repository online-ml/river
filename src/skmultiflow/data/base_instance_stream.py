from abc import ABCMeta, abstractmethod
from skmultiflow.core.base_object import BaseObject


class Stream(BaseObject, metaclass=ABCMeta):
    """ BaseInstanceStream
    
    The abstract class setting up the minimum requirements of a stream, 
    so that it can work along the other modules in the scikit-multiflow
    framework.
    
    Raises
    ------
    NotImplementedError: This is an abstract class.
    
    """

    @abstractmethod
    def estimated_remaining_instances(self):
        """ estimated_remaining_instances
        
        Returns the estimated number of remaining instances. Returns -1 
        if it's infinite (generators' case)
        
        Returns
        -------
        int
            Remaining number of instances.
        
        """
        raise NotImplementedError

    @abstractmethod
    def has_more_instances(self):
        raise NotImplementedError

    @abstractmethod
    def next_instance(self, batch_size=1):
        """ next_instance
        
        Generates or returns some amount of samples.
        
        Parameters
        ----------
        batch_size: int
            How many samples at a time to return.
        
        Returns
        -------
        tuple or tuple list
            A numpy.ndarray of shape (batch_size, n_features) and an array-like of size 
            n_classification_tasks, representing the next batch_size samples.
        
        """
        raise NotImplementedError

    @abstractmethod
    def is_restartable(self):
        raise NotImplementedError

    @abstractmethod
    def restart(self):
        """ restart
        
        Restart the stream's configurations.
        
        """
        raise NotImplementedError

    @abstractmethod
    def get_num_nominal_attributes(self):
        """ get_num_nominal_attributes
        
        Returns
        -------
        int
            The number of nominal attributes given by the stream.
        
        """
        raise NotImplementedError

    @abstractmethod
    def get_num_numerical_attributes(self):
        """ get_num_numerical_attributes

        Returns
        -------
        int
            The number of numerical attributes given by the stream.

        """
        raise NotImplementedError

    @abstractmethod
    def get_num_values_per_nominal_attribute(self):
        """ get_num_values_per_nominal_attribute

        Returns
        -------
        int
            The number of possible values for each nominal attribute.

        """
        raise NotImplementedError

    @abstractmethod
    def get_num_attributes(self):
        """ get_num_attributes

        Returns
        -------
        int
            The total number of attributes.

        """
        raise NotImplementedError

    @abstractmethod
    def get_num_classes(self):
        raise NotImplementedError

    @abstractmethod
    def get_attributes_header(self):
        raise NotImplementedError

    @abstractmethod
    def get_classes_header(self):
        raise NotImplementedError

    @abstractmethod
    def get_last_instance(self):
        raise NotImplementedError

    @abstractmethod
    def prepare_for_use(self):
        """ prepare_for_use
        
        Prepare the stream for use. Can be the reading of a file, or 
        the generation of a function, or anything necessary for the 
        stream to work after its initialization.
        
        Notes
        -----
        Every time a stream is created this function has to be called.
        
        """
        raise NotImplementedError

    @abstractmethod
    def get_plot_name(self):
        """ get_plot_name
        
        Gets the name of the plot, which is a string that will appear 
        in evaluation methods, to represent the stream.
        
        The default format is: 'Stream name - x labels'.
        
        Returns
        -------
        string
            A string representing the plot name.
        
        """
        raise NotImplementedError

    @abstractmethod
    def get_classes(self):
        """ get_classes
        
        Get all classes that can be generated, either by random generation 
        or by reading a file. In the latter, the classes can be interpreted 
        as a np.unique(y) where y is the labels matrix.
        
        :return: 
        """
        raise NotImplementedError

    def get_class_type(self):
        return 'stream'

    @abstractmethod
    def get_info(self):
        raise NotImplementedError

    @abstractmethod
    def get_num_outputs(self):
        raise NotImplementedError
