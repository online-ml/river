from abc import ABCMeta, abstractmethod
from skmultiflow.core.base_object import BaseObject


class Stream(BaseObject, metaclass=ABCMeta):
    """ The abstract class setting up the minimum requirements of a stream,
    so that it can work along the other modules in the scikit-multiflow
    framework.
    
    Raises
    ------
    NotImplementedError: This is an abstract class.
    
    """
    def __init__(self):
        self.n_samples = 0
        self.n_targets = 0
        self.n_features = 0
        self.n_num_features = 0
        self.n_cat_features = 0
        self.cat_features_idx = []
        self.features_labels = None
        self.outputs_labels = None
        self.current_sample_x = None
        self.current_sample_y = None
        self.sample_idx = 0

    @abstractmethod
    def n_remaining_samples(self):
        """ Returns the estimated number of remaining samples.
        
        Returns
        -------
        int
            Remaining number of samples. -1 if infinite (e.g. generator)
        
        """
        raise NotImplementedError

    @abstractmethod
    def has_more_samples(self):
        raise NotImplementedError

    @abstractmethod
    def next_sample(self, batch_size=1):
        """ Generates or returns next `batch_size` samples in the stream.
        
        Parameters
        ----------
        batch_size: int
            How many samples at a time to return.
        
        Returns
        -------
        tuple or tuple list
            A numpy.ndarray of shape (batch_size, n_features) and an array-like of size 
            n_targets, representing the next batch_size samples.

        """
        raise NotImplementedError

    @abstractmethod
    def last_sample(self):
        """ Retrieves last `batch_size` samples in the stream.

        Returns
        -------
        tuple or tuple list
            A numpy.ndarray of shape (batch_size, n_features) and an array-like of shape
            (batch_size, n_targets), representing the next batch_size samples.

        """
        raise NotImplementedError

    @abstractmethod
    def is_restartable(self):
        """ Determine if the stream is restartable. """
        raise NotImplementedError

    @abstractmethod
    def restart(self):
        """  Restart the stream. """
        raise NotImplementedError

    @abstractmethod
    def get_n_features(self):
        """ Retrieve the number of features.

        Returns
        -------
        int
            The total number of features.

        """
        raise NotImplementedError

    @abstractmethod
    def get_n_cat_features(self):
        """ Retrieve the number of nominal features.
        
        Returns
        -------
        int
            The number of nominal features in the stream.
        
        """
        raise NotImplementedError

    @abstractmethod
    def get_n_num_features(self):
        """ Retrieve the number of numerical atfeaturestributes.

        Returns
        -------
        int
            The number of numerical features in the stream.

        """
        raise NotImplementedError

    @abstractmethod
    def get_n_targets(self):
        raise NotImplementedError

    @abstractmethod
    def get_targets(self):
        """ Get all classes in the stream. """
        raise NotImplementedError

    @abstractmethod
    def get_feature_names(self):
        raise NotImplementedError

    @abstractmethod
    def get_target_names(self):
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
    def get_name(self):
        """ get_name
        
        Gets the name of the plot, which is a string that will appear 
        in evaluation methods, to represent the stream.
        
        The default format is: 'Stream name - x labels'.
        
        Returns
        -------
        string
            A string representing the plot name.
        
        """
        raise NotImplementedError

    def get_class_type(self):
        return 'stream'
