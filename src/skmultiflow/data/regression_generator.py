from skmultiflow.data.base_stream import Stream
from sklearn.datasets import make_regression
from skmultiflow.utils import check_random_state
import numpy as np


class RegressionGenerator(Stream):
    """ Creates a regression stream.
    
    This generator creates a stream of samples for a regression problem. It 
    uses the make_regression function from scikit-learn, which creates a 
    batch setting regression problem. These samples are then sequentially 
    fed by the next_sample function.
    
    Parameters
    ----------
    n_samples: int (Default: 40000)
        Total amount of samples to generate.
    
    n_features: int (Default: 100)
        Number of features to generate.
        
    n_informative: int (Default: 10)
        Number of relevant features, in other words, the number of features 
        that influence the class label.
    
    n_targets: int (Default: 1)
        Number of target_values (outputs) to generate.

    random_state: int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Notes
    -----
    This is a wrapper for scikit-lean's `make_regression`
    
    Examples
    --------
    >>> # Imports
    >>> from skmultiflow.data.regression_generator import RegressionGenerator
    >>> # Setting up the stream
    >>> stream = RegressionGenerator(n_samples=100, n_features=20, n_targets=4, n_informative=6, random_state=0)
    >>> stream.prepare_for_use()
    >>> # Retrieving one sample
    >>> stream.next_sample()
    (array([[ 0.16422776,  0.56729028, -0.76149221,  0.38728048, -1.69810582,
          0.85792392, -0.2226751 , -0.98551074,  1.46657872,  1.64813493,
          0.03863055,  1.14110187, -1.6567151 , -0.29183736, -1.02250684,
         -1.47183501, -1.61647419,  0.85255194, -2.25556423, -0.35343175]]),
         array([[-227.21175382, -208.69356686, -430.10330937, -439.69284148]]))
    >>> # Retrieving 10 samples
    >>> stream.next_sample(10)
    (array([[-0.30309825,  0.44103291,  0.41287082, -0.14456682,  0.3595044 ,
         -0.1983989 ,  0.17879287, -0.40594173, -1.14761094,  1.38526155,
         -0.93788023,  0.0941923 ,  0.43310795,  0.28912051,  1.06458514,
          0.7243685 ,  0.24078751, -0.35811408, -0.36159928, -0.7994224 ],
        [ 1.04297759,  0.41409135, -0.94893281,  0.16464381,  1.04008625,
          0.13191176, -0.50723446, -0.32656098,  0.76877064, -0.52261942,
          0.38909397, -1.98056559,  1.17104106, -0.03926799,  1.47376482,
         -0.00820988,  1.04156839, -0.42132759,  0.88518754,  0.15466883],
        [-0.83912419, -1.01177408,  0.75746833, -0.6432576 ,  1.58776152,
         -0.01005647,  0.08496814, -0.0451133 , -1.04059923,  0.85053068,
         -0.14876615,  1.23800694,  0.0960042 ,  1.86668315,  0.99675964,
          0.07912172, -1.37305354, -0.31560312, -1.13359283, -1.60643969],
        [ 0.9508337 ,  0.55929898,  1.30718385, -1.64134861,  1.39053397,
         -0.46744101, -1.06369559, -0.33868219,  0.85910419,  1.05417791,
         -0.49579549, -0.86015338,  1.21657771,  0.67755703,  0.06606026,
          2.03476254,  0.57275137, -0.80962658, -0.15503581, -0.43109634],
        [-0.80149689, -0.64718143,  1.99795608, -0.96460642,  1.32646164,
         -0.85654931,  0.47224715,  0.93639854,  2.59442459,  0.27117018,
         -0.76211451, -1.5415874 , -0.88778014, -1.42191987, -0.21252304,
         -0.52564059, -0.1753164 , -0.40403229,  0.05989468,  0.9304085 ],
        [-0.21120598, -0.12040664, -1.74418776,  0.87569568, -0.46931074,
          1.66060756, -1.47931598,  1.02122474, -2.8022028 ,  2.45122972,
         -0.48024204, -1.41660348, -0.52325094, -0.44876701,  1.94709864,
          0.70869527, -0.7214313 , -1.18842442, -1.36516288, -0.33210228],
        [ 0.49949823, -0.06205313,  1.76992139, -0.03093626, -1.1046166 ,
         -0.16821422,  1.25916713,  0.26902407,  1.32435875,  1.26741165,
         -0.56643985,  0.3779101 , -0.30769128,  2.52636824, -0.79550055,
          0.52491786, -1.49567952, -0.17220079,  1.57886519,  0.70411102],
        [ 0.8640523 , -2.23960406, -0.5854312 , -0.91307922, -0.22260568,
         -0.26164545,  0.40149906,  0.93674246, -0.20289684, -2.36958691,
          0.24211796, -0.18224478, -0.88872026, -1.27968917, -0.88897136,
          1.41232771,  0.06485611, -0.10988278, -1.68121822,  1.22487056],
        [ 0.61645931,  0.53659652,  0.08595197, -1.96273201, -0.89636972,
          0.75194659,  0.40469546,  0.87096178, -1.19498681,  1.29614987,
         -1.13900819,  0.56298972, -1.21440138, -0.45408036,  0.64796779,
         -0.87797062,  0.8805112 , -0.50040967,  1.58482053,  0.19145087],
        [ 1.30184623, -0.62808756,  1.13689136,  1.02017271, -0.11054066,
          0.09772497, -0.48102712, -1.04525337, -0.39944903,  0.68981816,
          0.28634369,  0.58295368,  0.60884383, -0.1359497 ,  1.53637705,
          1.21114529, -1.06001582,  0.37005589, -0.69204985,  2.3039167 ]]),
          array([[  31.59103587,   19.35028127,   33.49418263,   22.27335009],
          [ 153.04501993,  245.02067196,  338.82484458,  365.47183945],
          [  43.14398252,   47.75322041,    1.17298222,   44.35274394],
          [  93.58627672,  -65.01446316,   79.20394868,   46.55266948],
          [  -9.74401621, -137.01970244, -144.66863494, -123.09407564],
          [ -51.78237536,  103.64689371,  -37.00451143,  -15.08925677],
          [ -32.06049627, -127.04540624,  -21.14164295,  -80.71667   ],
          [-121.50880042, -197.05839429, -278.61694828, -291.47192161],
          [ -72.53226633, -280.00028587,  -44.57428097, -166.31003398],
          [  41.74351609,  220.43038917,  151.95222469,  182.65729147]]))

    >>> stream.n_remaining_samples()
    89
    >>> stream.has_more_samples()
    True
    
    """

    def __init__(self, n_samples=40000, n_features=100, n_informative=10, n_targets=1, random_state=None):
        super().__init__()
        self.X = None
        self.y = None
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_targets = n_targets
        self.n_informative = n_informative
        self.n_num_features = n_features
        self.n_features = n_features
        self.random_state = random_state
        self._random_state = None   # This is the actual random_state object used internally
        self.name = "Regression Generator"

    def prepare_for_use(self):
        """ Prepare the stream for usage
        
        Uses the make_regression function from scikit-learn to generate a 
        regression problem. This problem will be kept in memory and provided 
        as demanded.
        
        """
        self._random_state = check_random_state(self.random_state)
        self.X, self.y = make_regression(n_samples=self.n_samples,
                                         n_features=self.n_features,
                                         n_informative=self.n_informative,
                                         n_targets=self.n_targets,
                                         random_state=self._random_state)
        self.y = np.resize(self.y, (self.y.size, self.n_targets))

        self.target_names = ["target_" + str(i) for i in range(self.n_targets)]
        self.feature_names = ["att_num_" + str(i) for i in range(self.n_num_features)]
        self.target_values = [float] * self.n_targets

    def n_remaining_samples(self):
        """
        Returns
        -------
        int
            Number of samples remaining.
        """
        return self.n_samples - self.sample_idx

    def has_more_samples(self):
        """
        Returns
        -------
        Boolean
            True if stream has more samples.
        """
        return self.n_samples - self.sample_idx > 0

    def next_sample(self, batch_size=1):
        """ next_sample
        
        Returns batch_size samples from the generated regression problem.
        
        Parameters
        ----------
        batch_size: int
            The number of sample to return.
            
        Returns
        -------
        tuple or tuple list
            Return a tuple with the features matrix and the labels matrix for 
            the batch_size samples that were requested.
        
        """
        self.sample_idx += batch_size
        try:

            self.current_sample_x = self.X[self.sample_idx - batch_size:self.sample_idx, :]
            self.current_sample_y = self.y[self.sample_idx - batch_size:self.sample_idx, :]
            if self.n_targets < 2:
                self.current_sample_y = self.current_sample_y.flatten()

        except IndexError:
            self.current_sample_x = None
            self.current_sample_y = None
        return self.current_sample_x, self.current_sample_y

    def restart(self):
        """
        Restart the stream to the initial state.

        """
        # Note: No need to regenerate the data, just reset the idx
        self.sample_idx = 0
        self.current_sample_x = None
        self.current_sample_y = None

    def get_data_info(self):
        return "Regression Generator - {} targets, {} features".format(self.n_targets, self.n_features)
