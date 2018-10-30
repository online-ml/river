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
        self.n_classes = 0
        self.cat_features_idx = []
        self.current_sample_x = None
        self.current_sample_y = None
        self.sample_idx = 0
        self.feature_names = None
        self.target_names = None
        self.target_values = None
        self.name = None

    @property
    def n_features(self):
        """ Retrieve the number of features.

        Returns
        -------
        int
            The total number of features.

        """
        return self._n_features

    @n_features.setter
    def n_features(self, n_features):
        """ Set the number of features

        """
        self._n_features = n_features

    @property
    def n_cat_features(self):
        """ Retrieve the number of integer features.

        Returns
        -------
        int
            The number of integer features in the stream.

        """
        return self._n_cat_features

    @n_cat_features.setter
    def n_cat_features(self, n_cat_features):
        """ Set the number of integer features

        Parameters
        ----------
        n_cat_features: int
        """
        self._n_cat_features = n_cat_features

    @property
    def n_num_features(self):
        """ Retrieve the number of numerical features.

        Returns
        -------
        int
            The number of numerical features in the stream.

        """
        return self._n_num_features

    @n_num_features.setter
    def n_num_features(self, n_num_features):
        """ Set the number of numerical features

        Parameters
        ----------
        n_num_features: int

        """
        self._n_num_features = n_num_features

    @property
    def n_targets(self):
        """ Retrieve the number of targets

        Returns
        -------
        int
            the number of targets in the stream.
        """
        return self._target_idx

    @n_targets.setter
    def n_targets(self, n_targets):
        """ Set the number of targets

        Parameters
        ----------
        n_targets: int
        """
        self._target_idx = n_targets

    @property
    def target_values(self):
        """ Retrieve all target_values in the stream for each target.

        Returns
        -------
        list
            list of lists of all target_values for each target
        """
        return self._target_values

    @target_values.setter
    def target_values(self, target_values):
        """ Set the list for all target_values in the stream.

        Parameters
        ----------
        target_values
        """
        self._target_values = target_values

    @property
    def feature_names(self):
        """ Retrieve the names of the features.

        Returns
        -------
        list
            names of the features
        """
        return self._feature_names

    @feature_names.setter
    def feature_names(self, feature_names):
        """ Set the name of the features in the stream.

        Parameters
        ----------
        feature_names: list
        """
        self._feature_names = feature_names

    @property
    def target_names(self):
        """ Retrieve the names of the targets

        Returns
        -------
        list
            the names of the targets in the stream.
        """
        return self._target_names

    @target_names.setter
    def target_names(self, target_names):
        """ Set the names of the targets in the stream.

        Parameters
        ----------
        target_names: list

        """
        self._target_names = target_names

    @property
    def random_state(self):
        """ Retrieve the random state of the stream.

        Returns
        -------
        RandomState
        """
        return self._random_state

    @random_state.setter
    def random_state(self, random_state):
        """ Set the random state of the stream

        Parameters
        ----------
        random_state

        Returns
        -------

        """
        self._random_state = random_state

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

    def last_sample(self):
        """ Retrieves last `batch_size` samples in the stream.

        Returns
        -------
        tuple or tuple list
            A numpy.ndarray of shape (batch_size, n_features) and an array-like of shape
            (batch_size, n_targets), representing the next batch_size samples.

        """
        return self.current_sample_x, self.current_sample_y

    def is_restartable(self):
        """ Determine if the stream is restartable.
         Returns
         -------
         Boolean
            True if stream is restartable.
         """
        return True

    def restart(self):
        """  Restart the stream. """
        self.prepare_for_use()

    def n_remaining_samples(self):
        """ Returns the estimated number of remaining samples.

        Returns
        -------
        int
            Remaining number of samples. -1 if infinite (e.g. generator)

        """
        return -1

    def has_more_samples(self):
        """
        Checks if stream has more samples.

        Returns
        -------
        Boolean
            True if stream has more samples.
        """
        return True

    def get_data_info(self):
        """ get_name
        
        Gets the name of the plot, which is a string that will appear 
        in evaluation methods, to represent the stream.
        
        The default format is: 'Stream name - x labels'.
        
        Returns
        -------
        string
            A string representing the plot name.
        
        """
        return self.name + " - {} targets, {} classes, {} features".format(self.n_targets,
                                                                             self.n_classes, self.n_features)

    def get_class_type(self):
        return 'stream'
