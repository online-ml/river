from skmultiflow.data.base_stream import Stream
from skmultiflow.data import pseudo_random_processes as prp
from skmultiflow.utils import check_random_state
import numpy as np


class RandomRBFGenerator(Stream):
    """ Random Radial Basis Function stream generator.
    
    Produces a radial basis function stream. A number of centroids, having a random central position, a standard
    deviation, a class label and weight, are generated. A new sample is created by choosing one of the centroids at
    random, taking into account their weights, and offsetting the attributes at a random direction from the centroid's
    center. The offset length is drawn  from a Gaussian distribution.
      
    This process will create a normally distributed hypersphere of samples on the surrounds of each centroid.
    
    Parameters
    ---------
    model_random_state: int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`..

    sample_random_state: int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`..
        
    n_classes: int (Default: 2)
        The number of class labels to generate.
        
    n_features: int (Default: 10)
        The number of numerical features to generate.
        
    n_centroids: int (Default: 50)
        The number of centroids to generate.
        
    Examples
    --------
    >>> # Imports
    >>> from skmultiflow.data.random_rbf_generator import RandomRBFGenerator
    >>> # Setting up the stream
    >>> stream = RandomRBFGenerator(model_random_state=99, sample_random_state=50, n_classes=4, n_features=10, n_centroids=50)
    >>> stream.prepare_for_use()
    >>> # Retrieving one sample
    >>> stream.next_sample()
    (array([[ 0.44952282,  1.09201096,  0.34778443,  0.92181679,  0.19503463,
         0.28834419,  0.8293168 ,  0.26847952,  0.8096243 ,  0.23850379]]), array([[ 3.]]))
    >>> # Retrieving 10 samples
    >>> stream.next_sample(10)
    (array([[ 0.70374896,  0.65752835,  0.20343463,  0.56136917,  0.76659286,
         0.61081231,  0.60453064,  0.88734577, -0.04244631,  0.09146432],
       [ 0.27797196,  0.05640135,  0.80946171,  0.60572837,  0.95080656,
         0.25512099,  0.73992469,  0.33917142,  0.17104577,  0.79283295],
       [ 0.33696167,  0.10923638,  0.85987231,  0.61868598,  0.85755211,
         0.19469184,  0.66750447,  0.27684404,  0.1554274 ,  0.76262286],
       [ 0.71886223,  0.23078927,  0.45013806,  0.03019141,  0.42679505,
         0.03841721,  0.34318517,  0.11769923,  0.9644654 ,  0.01635577],
       [-0.01849262,  0.92570731,  0.87564868,  0.49372553,  0.39717634,
         0.46697609,  0.41329831,  0.27652149,  0.12724455,  0.24658299],
       [ 0.81850217,  0.87228851,  0.18873385, -0.04254749,  0.06942877,
         0.55567756,  0.97660009,  0.0273206 ,  0.67995834,  0.49135807],
       [ 0.69888163,  0.61994977,  0.43074298,  0.27526838,  0.69566798,
         0.91059369,  0.04680901,  0.50453698,  0.61394089,  0.92275292],
       [ 1.01929588,  0.80181051,  0.50547533,  0.14715636,  0.42889167,
         0.61513174,  0.21752655, -0.52958207,  1.35091672,  0.38769673],
       [ 0.37738633,  0.60922205,  0.64216064,  0.90009707,  0.91787083,
         0.36189554,  0.35438165,  0.28510134,  0.55301333,  0.21450072],
       [ 0.62185359,  0.75178244,  1.00436662,  0.24412816,  0.41070861,
         0.52547739,  0.50978735,  0.79445216,  0.77589569,  0.16214271]]), array([[ 3.],
       [ 3.],
       [ 3.],
       [ 2.],
       [ 3.],
       [ 2.],
       [ 0.],
       [ 2.],
       [ 0.],
       [ 2.]]))
    >>> # Generators will have infinite remaining instances, so it returns -1
    >>> stream.n_remaining_samples()
    -1
    >>> stream.has_more_samples()
    True
    
    """

    def __init__(self, model_random_state=None, sample_random_state=None, n_classes=2, n_features=10, n_centroids=50):
        super().__init__()
        self.sample_random_state = sample_random_state
        self.model_random_state = model_random_state
        self._sample_random_state = None   # This is the actual random_state object used internally
        self.n_classes = n_classes
        self.n_targets = 1
        self.n_features = n_features
        self.n_num_features = n_features
        self.n_centroids = n_centroids
        self.centroids = None
        self.centroid_weights = None
        self.name = "Random RBF Generator"

        self.__configure()

    def __configure(self):

        self.target_names = ["target_0"]
        self.feature_names = ["att_num_" + str(i) for i in range(self.n_num_features)]
        self.target_values = [i for i in range(self.n_classes)]

    def next_sample(self, batch_size=1):

        """ Return batch_size samples generated by choosing a centroid at
        random and randomly offsetting its attributes so that it is 
        placed inside the hypersphere of that centroid.
        
        Parameters
        ----------
        batch_size: int
            The number of samples to return.
        
        Returns
        -------
        tuple or tuple list
            Return a tuple with the features matrix and the labels matrix for 
            the batch_size samples that were requested. 
        
        """
        data = np.zeros([batch_size, self.n_features + 1])
        for j in range(batch_size):
            centroid_aux = self.centroids[prp.random_index_based_on_weights(self.centroid_weights,
                                                                            self._sample_random_state)]
            att_vals = []
            magnitude = 0.0
            for i in range(self.n_features):
                att_vals.append((self._sample_random_state.rand() * 2.0) - 1.0)
                magnitude += att_vals[i] * att_vals[i]
            magnitude = np.sqrt(magnitude)
            desired_mag = self._sample_random_state.normal() * centroid_aux.std_dev
            scale = desired_mag / magnitude
            for i in range(self.n_features):
                data[j, i] = centroid_aux.centre[i] + att_vals[i] * scale
            data[j, self.n_features] = centroid_aux.class_label
        self.current_sample_x = data[:, :self.n_features]
        self.current_sample_y = data[:, self.n_features:].flatten().astype(int)
        return self.current_sample_x, self.current_sample_y

    def prepare_for_use(self):
        """
        Prepares the stream for use.

        Notes
        -----
        This functions should always be called after the stream initialization.

        """
        self.generate_centroids()
        self._sample_random_state = check_random_state(self.sample_random_state)

    def generate_centroids(self):
        """ generate_centroids
        
        Sequentially creates all the centroids, choosing at random a center, 
        a label, a standard deviation and a weight. 
        
        """
        model_random_state = check_random_state(self.model_random_state)
        self.centroids = []
        self.centroid_weights = []
        for i in range(self.n_centroids):
            self.centroids.append(Centroid())
            rand_centre = []
            for j in range(self.n_num_features):
                rand_centre.append(model_random_state.rand())
            self.centroids[i].centre = rand_centre
            self.centroids[i].class_label = model_random_state.randint(self.n_classes)
            self.centroids[i].std_dev = model_random_state.rand()
            self.centroid_weights.append(model_random_state.rand())


class Centroid:
    """ Class that stores a centroid's attributes. """
    def __init__(self):
        self.centre = None
        self.class_label = None
        self.std_dev = None
