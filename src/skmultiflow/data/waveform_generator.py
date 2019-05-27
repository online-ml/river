import numpy as np
from skmultiflow.data.base_stream import Stream
from skmultiflow.utils import check_random_state


class WaveformGenerator(Stream):
    """ Waveform stream generator.

    Generates instances with 21 numeric attributes and 3 classes, based
    on a random differentiation of some base waveforms. Supports noise 
    addition, but in this case the generator will have 40 attribute 
    instances
     
    Parameters
    ----------
    random_state: int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    has_noise: bool
        if True additional 19 unrelated features will be added. (Default: False)
        
    Examples
    --------
    >>> # Imports
    >>> from skmultiflow.data.waveform_generator import WaveformGenerator
    >>> # Setting up the stream
    >>> stream = WaveformGenerator(random_state=774, has_noise=True)
    >>> stream.prepare_for_use()
    >>> # Retrieving one sample
    >>> stream.next_sample()
    (array([[ -3.87277692e-03,   5.35892351e-01,  -6.07354638e-02,
          1.70731601e+00,   5.34361689e-01,  -1.77051944e-01,
          1.14121806e+00,   1.35608518e-01,   1.41239266e+00,
          3.54064724e+00,   3.07032776e+00,   4.51698567e+00,
          4.68043133e+00,   3.56075018e+00,   3.83788037e+00,
          2.71987164e+00,   4.77706723e-01,   2.12187988e+00,
          1.59313816e+00,  -5.11689592e-01,   5.99317674e-01,
          2.14508816e-01,  -1.05534090e+00,  -1.34679419e-01,
          5.32610078e-01,  -1.39251668e+00,   1.13220325e+00,
          3.04748552e-01,   1.41454012e+00,   6.73651106e-01,
          1.85981832e-01,  -1.76774471e+00,   3.31777766e-02,
          8.17011922e-02,   1.70686324e+00,   1.10471095e+00,
         -5.08874759e-01,   4.16279862e-01,  -4.26805543e-01,
          9.94596567e-01]]), array([ 2.]))
    >>> # Retrieving 2 samples
    >>> stream.next_sample(2)
    (array([[ -6.72385828e-01,   1.51039782e+00,   5.64599422e-01,
          2.77481410e+00,   2.27653785e+00,   4.40016488e+00,
          3.87856303e+00,   4.90321750e+00,   4.40651078e+00,
          5.07337409e+00,   3.23727692e+00,   2.99724461e+00,
          1.46384329e+00,   1.30042173e+00,   3.67083253e-02,
          3.80546239e-01,  -2.05337011e+00,   6.06889589e-01,
         -1.10649679e+00,   3.38098465e-01,  -8.33683681e-01,
         -3.35283052e-02,  -6.65394037e-01,  -1.09290608e+00,
          4.15778821e-01,   3.64210364e-01,   1.18695445e+00,
          2.77980322e-01,   8.69224059e-01,  -4.93428014e-01,
         -1.08745643e+00,  -9.80906438e-01,   4.12116697e-01,
          2.39579703e-01,   1.53145126e+00,   6.26022691e-01,
          9.82996997e-02,   8.33911055e-01,   8.55830752e-02,
          1.54462877e+00],
       [  3.34017183e-01,  -5.00919347e-01,   2.67311051e+00,
          3.23473039e+00,   2.04091185e+00,   5.62868585e+00,
          4.79611194e+00,   4.14500688e+00,   5.76366418e+00,
          4.18105491e+00,   4.73064582e+00,   3.03461230e+00,
          1.79417942e+00,  -9.84100765e-01,   1.34212863e+00,
          1.29337991e-01,   6.08571939e-01,  -8.56504577e-01,
          2.95358587e-01,   9.12880505e-01,   2.88118824e-01,
         -4.49398914e-01,   5.44025828e-03,  -1.78535212e+00,
          1.41541455e-01,  -6.91216596e-01,  -8.66808201e-02,
         -1.27541907e-01,  -5.38038710e-01,  -1.19807563e+00,
          1.03113317e+00,   2.39999025e-01,   5.24084853e-02,
          1.04314518e+00,   3.20412032e+00,   1.26117112e+00,
         -7.10479419e-01,   4.60132194e-01,  -5.63210805e-02,
         -1.56883271e-01]]), array([ 1.,  1.]))
    >>> # Generators will have infinite remaining instances, so it returns -1
    >>> stream.n_remaining_samples()
    -1
    >>> stream.has_more_samples()
    True
        
    """
    
    _NUM_CLASSES = 3
    _NUM_BASE_ATTRIBUTES = 21
    _TOTAL_ATTRIBUTES_INCLUDING_NOISE = 40
    _H_FUNCTION = np.array([[0, 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 0],
                            [0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 0, 0, 0, 0, 0]])

    def __init__(self, random_state=None, has_noise=False):
        super().__init__()

        self.random_state = random_state
        self._random_state = None   # This is the actual random_state object used internally
        self.has_noise = has_noise
        self.n_num_features = self._NUM_BASE_ATTRIBUTES
        self.n_classes = self._NUM_CLASSES
        self.n_targets = 1
        self.name = "Waveform Generator"

        self.__configure()

    def __configure(self):

        if self.has_noise:
            self.n_num_features = self._TOTAL_ATTRIBUTES_INCLUDING_NOISE
        self.n_features = self.n_num_features
        self.feature_names = ["att_num_" + str(i) for i in range(self.n_features)]
        self.target_names = ["target_0"]
        self.target_values = [i for i in range(self.n_classes)]

    @property
    def has_noise(self):
        """ Retrieve the value of the option: add noise.

        Returns
        -------
        Boolean
            True is the noise is added.
        """
        return self._has_noise

    @has_noise.setter
    def has_noise(self, has_noise):
        """ Set the value of the option: add noise.

        Parameters
        ----------
        has_noise: Boolean

        """
        if isinstance(has_noise, bool):
            self._has_noise = has_noise
        else:
            raise ValueError("has_noise should be boolean, {} was passed".format(has_noise))

    def prepare_for_use(self):
        """
        Prepares the stream for use.

        Notes
        -----
        This functions should always be called after the stream initialization.

        """
        self._random_state = check_random_state(self.random_state)
        self.sample_idx = 0

    def next_sample(self, batch_size=1):
        """ next_sample
        
        An instance is generated based on the parameters passed. If noise 
        is included the total number of features will be 40, if it's not
        included there will be 21 attributes. In both cases there is one 
        classification task, which chooses one between three labels.
        
        After the number of attributes is chosen, the algorithm will randomly
        choose one of the hard coded waveforms, as well as random multipliers. 
        For each attribute, the actual value generated will be a a combination 
        of the hard coded functions, with the multipliers and a random value.
        
        Furthermore, if noise is added the features from 21 to 40 will be
        replaced with a random normal value.
        
        Parameters
        ----------
        batch_size: int
            The number of samples to return.
            
        Returns
        -------
        tuple or tuple list
            Return a tuple with the features matrix and the labels matrix 
            for the batch_size samples that were requested.
        
        """
        data = np.zeros([batch_size, self.n_features + 1])

        for j in range(batch_size):
            self.sample_idx += 1
            group = self._random_state.randint(0, self.n_classes)
            choice_a = 1 if (group == 2) else 0
            choice_b = 1 if (group == 0) else 2
            multiplier_a = self._random_state.rand()
            multiplier_b = 1.0 - multiplier_a

            for i in range(self._NUM_BASE_ATTRIBUTES):
                data[j, i] = multiplier_a * self._H_FUNCTION[choice_a][i] \
                            + multiplier_b * self._H_FUNCTION[choice_b][i] \
                            + self._random_state.normal()

            if self.has_noise:
                for i in range(self._NUM_BASE_ATTRIBUTES, self._TOTAL_ATTRIBUTES_INCLUDING_NOISE):
                    data[j, i] = self._random_state.normal()

            data[j, data[j].size - 1] = group
        self.current_sample_x = data[:, :self.n_features]
        self.current_sample_y = np.ravel(data[:, self.n_features:]).astype(int)

        return self.current_sample_x, self.current_sample_y
