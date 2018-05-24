import numpy as np
from skmultiflow.data.base_stream import Stream
from skmultiflow.core.utils.validation import check_random_state


class HyperplaneGenerator(Stream):
    """ HyperplaneGenerator

    Parameters
    ----------
    random_state: int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used by `np.random`.

    n_features: int (Default 10)
        The number of attributes to generate. ( from 2 to Max int)

    n_drift_features: int (Default: 2)
        The number of attributes with drift. ( from 2 to Max int)

    mag_change: float (Default: 0.0)
        Magnitude of the change for every example. (from  0.0 to 1.0)

    noise_percentage: float (Default: 0.05)
        Percentage of noise to add to the data. ( from 0.0 to 1.0)

    sigma_percentage: int (Default 0.1)
        Percentage of probability that the direction of change is reversed. (from 0.0 to 1.0)



    """

    def __init__(self, random_state=None, n_features=10, n_drift_features=2, mag_change=0.0,
                 noise_percentage=0.05, sigma_percentage=0.1):
        super().__init__()

        self._original_random_state = random_state
        self.n_num_features = n_features
        self.n_features = self.n_num_features
        self.n_classes = 2
        self.n_drift_features = n_drift_features
        self.mag_change = mag_change
        self.sigma_percentage = sigma_percentage
        self.noise_percentage = noise_percentage
        self.n_targets = 1
        self.random_state = None
        self._next_class_should_be_zero = False
        self._weights = np.zeros(self.n_features)
        self._sigma = np.zeros(self.n_features)
        self.name = "Hyperplane Generator"

        self.__configure()

    def __configure(self):
        self.target_names = ["target_0"]
        self.feature_names = ["att_num_" + str(i) for i in range(self.n_features)]
        self.target_values = [i for i in range(self.n_classes)]

    def prepare_for_use(self):
        self.random_state = check_random_state(self._original_random_state)
        self._next_class_should_be_zero = False
        self.sample_idx = 0
        for i in range(self.n_features):
            self._weights[i] = self.random_state.rand()
            self._sigma[i] = 1 if (i < self.n_drift_features) else 0

    def next_sample(self, batch_size=1):
        """ next_sample

        The sample generation works as follows: The n_features are generated
        with the random generator, initialized with the seed passed by the
        user. Then, the classification function decides, as a function of the
        sum and weight' sum, whether to classify the instance as class 0 or
        class 1. The next step is to generate drift.

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
        sum = 0
        sum_weights = 0.0

        for j in range(batch_size):
            self.sample_idx += 1
            for i in range(self.n_features):
                data[j, i] = self.random_state.rand()
                sum += self._weights[i] * data[j, i]
                sum_weights += self._weights[i]

            group = 1 if sum >= sum_weights * 0.5 else 0

            if 0.01 + self.random_state.rand() <= self.noise_percentage:
                group = 1 if (group == 0) else 0

            data[j, -1] = group

        self.generate_drift()

        self.current_sample_x = data[:, :self.n_features]
        self.current_sample_y = data[:, self.n_features:].flatten()

        return self.current_sample_x, self.current_sample_y

    def generate_drift(self):
        for i in range(self.n_drift_features):
            self._weights[i] += float(float(self._sigma[i]) * float(self.mag_change))
            if (0.01 + self.random_state.rand()) <= self.sigma_percentage:
                self._sigma[i] *= -1

    def get_info(self):
        return 'HyperplaneGenerator: - random_state: ' + str(self._original_random_state) + \
               ' - n_features: ' + str(self.n_features) + \
               ' - n_classes: ' + str(self.n_classes) + \
               ' - n_drift_features: ' + str(self.n_drift_features) + \
               ' - perturbation: ' + str(self.noise_percentage) + \
               ' - sigma_percentage: ' + str(self.sigma_percentage) + \
               ' - mag_change: ' + str(self.mag_change)

