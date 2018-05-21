import numpy as np
from skmultiflow.data.base_stream import Stream
from skmultiflow.core.utils.validation import check_random_state


class HyperplaneGenerator(Stream):
    """ HyperplaneGenerator

    Parameters
    ----------
    random_state: int (Default: None)
        The seed used to initialize the random generator, which is an instance
        of numpy's random.

    n_classes: int (Default: 2)
        The number of classes to generate.. (from 2 to Max int)

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

    Notes
    -----
    Number of classes is fixed to 2, support of more than 2 classes is not available yet.


    """

    def __init__(self, random_state=None, n_classes=2, n_features=10, n_drift_features=2, mag_change=0.0,
                 noise_percentage=0.05, sigma_percentage=0.1):
        super().__init__()

        self.random_state = random_state
        self.n_features = n_features
        self.n_classes = n_classes
        self.n_drift_features = n_drift_features
        self.mag_change = mag_change
        self.sigma_percentage = sigma_percentage
        self.noise_percentage = noise_percentage
        self.n_targets = 1
        self.sample_random = None
        self.next_class_should_be_zero = False
        self.weights = np.zeros(self.n_features)
        self.sigma = np.zeros(self.n_features)

        self.__configure()

    def __configure(self):
        self.sample_random = check_random_state(self.random_state)
        self.next_class_should_be_zero = False
        self.outputs_labels = []
        for i in range(self.n_classes):
            self.outputs_labels.append("class_num_" + str(i))
        self.features_labels = []
        for i in range(self.n_features):
            self.features_labels.append("att_num_" + str(i))
        for i in range(self.n_features):
            self.weights[i] = self.sample_random.rand()
            self.sigma[i] = 1 if (i < self.n_drift_features) else 0

    def n_remaining_samples(self):
        return -1

    def has_more_samples(self):
        return True

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
            for i in range(self.n_features):
                data[j, i] = self.sample_random.rand()
                sum += self.weights[i] * data[j, i]
                sum_weights += self.weights[i]

            group = 1 if sum >= sum_weights * 0.5 else 0

            if 0.01 + self.sample_random.rand() <= self.noise_percentage:
                group = 1 if (group == 0) else 0

            data[j, -1] = group

        self.generate_drift()

        self.current_sample_x = data[:, :self.n_features]
        self.current_sample_y = data[:, self.n_features:].flatten()

        return self.current_sample_x, self.current_sample_y

    def prepare_for_use(self):
        self.restart()

    def is_restartable(self):
        return True

    def restart(self):
        self.sample_random = check_random_state(self.random_state)
        self.next_class_should_be_zero = False

    def get_n_cat_features(self):
        return self.n_cat_features

    def get_n_num_features(self):
        return self.n_features

    def get_n_features(self):
        return self.n_features

    def get_n_targets(self):
        return self.n_targets

    def get_n_classes(self):
        return self.n_classes

    def get_feature_names(self):
        return self.features_labels

    def get_target_names(self):
        return self.outputs_labels

    def last_sample(self):
        return self.current_sample_x, self.current_sample_y

    def get_name(self):
        return "Hyperplane Generator - {} target, {} classes, {} features".format(self.n_targets, self.n_classes, self.n_features)

    def get_targets(self):
        return [i for i in range(self.n_classes)]

    def get_info(self):
        return 'HyperplaneGenerator: - random_state: ' + str(self.random_state) + \
               ' - n_features: ' + str(self.n_features) + \
               ' - n_classes: ' + str(self.n_classes) + \
               ' - n_drift_features: ' + str(self.n_drift_features) + \
               ' - perturbation: ' + str(self.noise_percentage) + \
               ' - sigma_percentage: ' + str(self.sigma_percentage) + \
               ' - mag_change: ' + str(self.mag_change)

    def generate_drift(self):
        for i in range(self.n_drift_features):
            self.weights[i] += self.sigma[i] * self.mag_change
            if (0.01 + self.sample_random.rand()) <= self.sigma_percentage:
                self.sigma[i] *= -1
