import numpy as np
from skmultiflow.data.base_stream import Stream
from skmultiflow.utils import check_random_state


class HyperplaneGenerator(Stream):
    r""" Hyperplane stream generator.

    Generates a problem of prediction class of a rotation hyperplane. It was
    used as testbed for CVFDT and VFDT in [1]_.

    A hyperplane in d-dimensional space is the set of points :math:`x` that satisfy
    :math:`\sum^{d}_{i=1} w_i x_i = w_0 = \sum^{d}_{i=1} w_i`, where
    :math:`x_i` is the ith coordinate of :math:`x`. Examples for which
    :math:`\sum^{d}_{i=1} w_i x_i > w_0`, are labeled positive, and examples
    for which :math:`\sum^{d}_{i=1} w_i x_i \leq w_0`, are labeled negative.

    Hyperplanes are useful for simulating time-changing concepts, because we
    can change the orientation and position of the hyperplane in a smooth
    manner by changing the relative size of the weights. We introduce change
    to this dataset by adding drift to each weight feature :math:`w_i = w_i + d \sigma`,
    where :math:`\sigma` is the probability that the direction of change is
    reversed and :math:`d` is the change applied to every example.

    Parameters
    ----------
    random_state: int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    n_features: int (Default 10)
        The number of attributes to generate.
        Higher than 2.

    n_drift_features: int (Default: 2)
        The number of attributes with drift.
        Higher than 2.

    mag_change: float (Default: 0.0)
        Magnitude of the change for every example.
        From 0.0 to 1.0.

    noise_percentage: float (Default: 0.05)
        Percentage of noise to add to the data.
        From 0.0 to 1.0.

    sigma_percentage: int (Default 0.1)
        Percentage of probability that the direction of change is reversed.
        From 0.0 to 1.0.

    References
    ----------
    .. [1] G. Hulten, L. Spencer, and P. Domingos. Mining time-changing data streams.
       In KDD’01, pages 97–106, San Francisco, CA, 2001. ACM Press.

    """

    def __init__(self, random_state=None, n_features=10, n_drift_features=2, mag_change=0.0,
                 noise_percentage=0.05, sigma_percentage=0.1):
        super().__init__()

        self.random_state = random_state
        self.n_num_features = n_features
        self.n_features = self.n_num_features
        self.n_classes = 2
        self.n_drift_features = n_drift_features
        self.mag_change = mag_change
        self.sigma_percentage = sigma_percentage
        self.noise_percentage = noise_percentage
        self.n_targets = 1
        self._random_state = None  # This is the actual random_state object used internally
        self._next_class_should_be_zero = False
        self._weights = np.zeros(self.n_features)
        self._sigma = np.zeros(self.n_features)
        self.name = "Hyperplane Generator"

        self.target_names = ["target_0"]
        self.feature_names = ["att_num_" + str(i) for i in range(self.n_features)]
        self.target_values = [i for i in range(self.n_classes)]

        self._prepare_for_use()

    @property
    def n_drift_features(self):
        """ Retrieve the number of drift features.

        Returns
        -------
        int
            The total number of drift features.

        """
        return self._n_drift_features

    @n_drift_features.setter
    def n_drift_features(self, n_drift_features):
        """ Set the number of drift features

        """
        self._n_drift_features = n_drift_features

    @property
    def noise_percentage(self):
        """ Retrieve the value of the value of Noise percentage

        Returns
        -------
        float
            percentage of the noise
        """
        return self._noise_percentage

    @noise_percentage.setter
    def noise_percentage(self, noise_percentage):
        """ Set the value of the value of noise percentage.

        Parameters
        ----------
        noise_percentage: float (0.0..1.0)

        """
        if (0.0 <= noise_percentage) and (noise_percentage <= 1.0):
            self._noise_percentage = noise_percentage
        else:
            raise ValueError(
                "noise percentage should be in [0.0..1.0], {} was passed".format(noise_percentage))

    @property
    def mag_change(self):
        """ Retrieve the value of the value of magnitude of change.

        Returns
        -------
        float
            magnitude of change
        """
        return self._mag_change

    @mag_change.setter
    def mag_change(self, mag_change):
        """ Set the value of the value of magnitude of change

        Parameters
        ----------
        mag_change: float (0.0..1.0)

        """
        if (0.0 <= mag_change) and (mag_change <= 1.0):
            self._mag_change = mag_change
        else:
            raise ValueError(
                "noise percentage should be in [0.0..1.0], {} was passed".format(mag_change))

    @property
    def sigma_percentage(self):
        """ Retrieve the value of the value of sigma percentage

        Returns
        -------
        float
            percentage of the sigma
        """
        return self._sigma_percentage

    @sigma_percentage.setter
    def sigma_percentage(self, sigma_percentage):
        """ Set the value of the value of noise percentage.

        Parameters
        ----------
        sigma_percentage: float (0.0..1.0)

        """
        if (0.0 <= sigma_percentage) and (sigma_percentage <= 1.0):
            self._sigma_percentage = sigma_percentage
        else:
            raise ValueError(
                "sigma percentage should be in [0.0..1.0], {} was passed".format(sigma_percentage))

    def _prepare_for_use(self):
        self._random_state = check_random_state(self.random_state)
        self._next_class_should_be_zero = False
        for i in range(self.n_features):
            self._weights[i] = self._random_state.rand()
            self._sigma[i] = 1 if (i < self.n_drift_features) else 0

    def next_sample(self, batch_size=1):
        """ Returns next sample from the stream.

        The sample generation works as follows: The features are generated
        with the random generator, initialized with the seed passed by the
        user. Then the classification function decides, as a function of the
        sum and weight's sum, whether to instance belongs to class 0 or
        class 1. The next step is to add noise if requested by the user and
        than generate drift.

        Parameters
        ----------
        batch_size: int (optional, default=1)
            The number of samples to return.

        Returns
        -------
        tuple or tuple list
            Return a tuple with the features matrix and the labels matrix for
            the batch_size samples that were requested.

        """
        data = np.zeros([batch_size, self.n_features + 1])

        for j in range(batch_size):
            sum_weights = np.sum(self._weights)
            self.sample_idx += 1
            sum = 0
            for i in range(self.n_features):
                data[j, i] = self._random_state.rand()
                sum += self._weights[i] * data[j, i]

            group = 1 if sum >= sum_weights * 0.5 else 0

            if 0.01 + self._random_state.rand() <= self.noise_percentage:
                group = 1 if (group == 0) else 0

            data[j, -1] = group

            self._generate_drift()

        self.current_sample_x = data[:, :self.n_features]
        self.current_sample_y = data[:, self.n_features:].flatten().astype(int)

        return self.current_sample_x, self.current_sample_y

    def _generate_drift(self):
        """
        Generate drift in the stream.

        """
        for i in range(self.n_drift_features):
            self._weights[i] += float(float(self._sigma[i]) * float(self.mag_change))
            if (0.01 + self._random_state.rand()) <= self.sigma_percentage:
                self._sigma[i] *= -1
