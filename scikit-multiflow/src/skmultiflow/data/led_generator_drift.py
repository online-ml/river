import numpy as np
from skmultiflow.data.led_generator import LEDGenerator


class LEDGeneratorDrift(LEDGenerator):
    """ LED stream generator with concept drift.

    This class is an extension from the LEDGenerator. The purpose of this generator is to
    add concept drift to the stream.

    Parameters
    ----------
    random_state: int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used by `np.random`.

    noise_percentage: float (Default: 0.0)
        The probability that noise will happen in the generation. At each
        new sample generated, a random probability is generated, and if that
        probability is equal or less than the noise_percentage, the selected data  will
        be switched.

    has_noise: bool (Default: False)
        Adds 17 non relevant attributes to the stream.

    n_drift_features : int (Default : 0)
        The number of attributes that have drift.

    Examples
    --------
       >>> # Imports
       >>> from skmultiflow.data.led_generator_drift import LEDGeneratorDrift
       >>> # Setting up the stream
       >>> stream = LEDGeneratorDrift(random_state = 112, noise_percentage = 0.28,has_noise= True,
       ... n_drift_features=4)
       >>> # Retrieving one sample
       >>> stream.next_sample()
       (array([[0., 1., 1., 1., 0., 1., 1., 0., 1., 0., 0., 0., 1., 0., 1., 1.,
         1., 0., 0., 0., 0., 0., 1., 1.]]), array([4]))

       >>> # Retrieving 10 samples
       >>> stream.next_sample(10)
       (array([[0., 0., 1., 0., 1., 0., 0., 1., 0., 0., 1., 1., 0., 0., 0., 0.,
         1., 1., 0., 0., 0., 0., 1., 1.],
        [0., 1., 1., 0., 0., 0., 1., 1., 1., 0., 1., 0., 0., 0., 1., 1.,
         1., 1., 1., 0., 1., 1., 1., 0.],
        [1., 1., 1., 0., 0., 1., 1., 1., 0., 0., 0., 0., 1., 0., 0., 0.,
         0., 1., 0., 1., 1., 0., 1., 1.],
        [0., 1., 0., 0., 1., 0., 0., 1., 0., 1., 1., 0., 1., 1., 0., 0.,
         1., 1., 0., 1., 1., 1., 1., 0.],
        [0., 1., 1., 0., 1., 0., 1., 0., 1., 1., 0., 1., 1., 0., 1., 0.,
         0., 0., 0., 1., 0., 1., 0., 0.],
        [1., 1., 1., 0., 1., 0., 1., 0., 1., 1., 0., 1., 0., 1., 1., 1.,
         0., 0., 0., 1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0.,
         1., 1., 1., 0., 1., 0., 0., 1.],
        [1., 0., 0., 0., 1., 1., 0., 1., 1., 1., 0., 0., 0., 0., 0., 1.,
         1., 1., 0., 1., 0., 0., 1., 1.],
        [0., 1., 1., 0., 1., 0., 0., 1., 1., 1., 0., 1., 1., 1., 1., 1.,
         0., 1., 0., 1., 0., 1., 0., 1.],
        [0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
         1., 1., 0., 1., 1., 1., 1., 0.]]),
        array([1, 0, 7, 9, 7, 1, 3, 1, 4, 1]))

       >>> # Generators will have infinite remaining instances, so it returns -1
       >>> stream.n_remaining_samples()
        -1
       >>> stream.has_more_samples()
        True

    """

    _numberAttribute = np.zeros((24,), dtype=int)
    _NUM_IRRELEVANT_ATTRIBUTES = 17

    def __init__(self, random_state=None, noise_percentage=0.0, has_noise=False,
                 n_drift_features=0):
        super().__init__(random_state=random_state, noise_percentage=noise_percentage,
                         has_noise=has_noise)
        self.n_drift_features = n_drift_features
        self.name = "Led Generator with drift"

        for i in range(self._TOTAL_ATTRIBUTES_INCLUDING_NOISE):
            self._numberAttribute[i] = i

        if self.has_noise and self.n_drift_features > 0:
            random_int = self._random_state.randint(7)
            offset = self._random_state.randint(self._NUM_IRRELEVANT_ATTRIBUTES)
            for i in range(self.n_drift_features):
                value1 = (i + random_int) % 7
                value2 = 7 + (i + offset) % self._NUM_IRRELEVANT_ATTRIBUTES
                self._numberAttribute[value1] = value2
                self._numberAttribute[value2] = value1

        self._prepare_for_use()

    def next_sample(self, batch_size=1):
        """ Returns next sample from the stream.

        An instance is generated based on the parameters passed. If noise
        is included the total number of attributes will be 24, if it's not
        included there will be 7 attributes.

        Parameters
        ----------
        batch_size: int (optional, default=1)
            The number of samples to return.

        Returns
        -------
        tuple or tuple list
            Return a tuple with the features matrix
            for the batch_size samples that were requested.

        """
        data = np.zeros([batch_size, self.n_features + 1])
        target = np.zeros(batch_size, dtype=int)

        for j in range(batch_size):
            self.sample_idx += 1
            selected = self._random_state.randint(self.n_classes)
            target[j] = selected
            for i in range(self._NUM_BASE_ATTRIBUTES):
                if (0.01 + self._random_state.rand()) <= self.noise_percentage:
                    data[j, self._numberAttribute[i]] = 1 if (
                        self._ORIGINAL_INSTANCES[selected, i] == 0) else 0
                else:
                    data[j, self._numberAttribute[i]] = self._ORIGINAL_INSTANCES[selected, i]
            if self.has_noise:
                for i in range(self._NUM_BASE_ATTRIBUTES, self._TOTAL_ATTRIBUTES_INCLUDING_NOISE):
                    data[j, self._numberAttribute[i]] = self._random_state.randint(2)

        self.current_sample_x = data[:, :self.n_features]
        self.current_sample_y = target
        return self.current_sample_x, self.current_sample_y

    def get_data_info(self):
        return "Led Generator with drift - {} features".format(self.n_features)
