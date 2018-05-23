import numpy as np
from skmultiflow.data.base_stream import Stream
from skmultiflow.core.utils.validation import check_random_state


class LEDGenerator(Stream):
    """ LEDGenerator

       Generates instances with 7 numeric attributes  Supports noise
       addition, but in this case the generator will have 24 attribute
       instances.
       The goal is to predict the digit displayed on a seven-segment
       LED display,

       Parameters
       ----------
       seed: int
           random_state for random generation of instances (Default: None)

       noise_percentage: float (Default: 0.0)
           The probability that noise will happen in the generation. At each
           new sample generated, a random probability is generated, and if that
           probability is equal or less than the noise_percentage, the selected data  will
           be switched

       add_noise: bool (Default: False)
           Adds 17 non relevant attributes to the stream.

       Examples
       --------
       >>> # Imports
       >>> from skmultiflow.data.generators.led_generator import LEDGenerator
       >>> # Setting up the stream
       >>> stream = LEDGenerator(seed = 112, noise_percentage = 0.28, add_noise= True)
       >>> stream.prepare_for_use()
       >>> # Retrieving one sample
       >>> stream.next_sample()
       array([[0., 1., 1., 1., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 1., 1.,
        1., 0., 0., 1., 1., 0., 1., 1.]])

       >>> # Retrieving 10 samples
       >>> stream.next_sample(10)
        array([[0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 0., 0.,
        1., 1., 1., 0., 0., 0., 1., 1.],
       [1., 1., 1., 0., 1., 0., 1., 1., 1., 0., 1., 0., 0., 0., 1., 1.,
        1., 1., 0., 0., 1., 0., 1., 0.],
       [0., 1., 1., 0., 0., 1., 1., 1., 0., 0., 0., 0., 1., 0., 0., 0.,
        0., 1., 0., 1., 1., 1., 1., 1.],
       [1., 1., 0., 0., 0., 1., 1., 1., 0., 1., 1., 0., 1., 1., 0., 0.,
        1., 1., 1., 0., 0., 0., 1., 0.],
       [1., 1., 1., 0., 0., 1., 0., 0., 1., 1., 0., 1., 1., 0., 1., 0.,
        0., 0., 1., 0., 1., 0., 0., 0.],
       [0., 1., 1., 0., 0., 1., 0., 0., 1., 1., 0., 1., 0., 1., 1., 1.,
        0., 0., 1., 0., 1., 1., 0., 0.],
       [0., 0., 0., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0.,
        1., 1., 1., 0., 1., 0., 0., 1.],
       [0., 0., 0., 0., 0., 1., 0., 1., 1., 1., 0., 0., 0., 0., 0., 1.,
        1., 1., 1., 1., 0., 1., 1., 1.],
       [1., 1., 1., 0., 0., 1., 0., 1., 1., 1., 0., 1., 1., 1., 1., 1.,
        0., 1., 1., 0., 0., 0., 0., 1.],
       [1., 1., 1., 0., 0., 1., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
        1., 1., 0., 0., 0., 0., 1., 0.]])

       >>> # Generators will have infinite remaining instances, so it returns -1
       >>> stream.n_remaining_samples()
      -1
       >>> stream.has_more_samples()
      True


    """
    _NUM_BASE_ATTRIBUTES = 7
    _TOTAL_ATTRIBUTES_INCLUDING_NOISE = 24
    _ORIGINAL_INSTANCES = np.array([[1, 1, 1, 0, 1, 1, 1],
                                   [0, 0, 1, 0, 0, 1, 0],
                                   [1, 0, 1, 1, 1, 0, 1],
                                   [1, 0, 1, 1, 0, 1, 1],
                                   [0, 1, 1, 1, 0, 1, 0],
                                   [1, 1, 0, 1, 0, 1, 1],
                                   [1, 1, 0, 1, 1, 1, 1],
                                   [1, 0, 1, 0, 0, 1, 0],
                                   [1, 1, 1, 1, 1, 1, 1],
                                   [1, 1, 1, 1, 0, 1, 1]])

    def __init__(self, seed=None, noise_percentage=0.0, add_noise=False):
        super().__init__()
        self._original_seed = seed
        self.random_state = None
        self.noise_percentage = noise_percentage
        self.n_features = self._NUM_BASE_ATTRIBUTES
        self.n_cat_features = self.n_features
        self.add_noise = add_noise
        self.n_targets = 0
        self.random_state = None
        self.__configure()

    def __configure(self):
        self.random_state = check_random_state(self._original_seed)
        self.n_features = self._TOTAL_ATTRIBUTES_INCLUDING_NOISE if self.has_noise() else self._NUM_BASE_ATTRIBUTES
        self.n_cat_features = self.n_features
        self.feature_names = ["att_num_" + str(i) for i in range(self.n_cat_features)]
        self.target_names = ["class"]
        self.targets = [i for i in range(self.n_targets)]

    def prepare_for_use(self):
        self.random_state = check_random_state(self._original_seed)

    def n_remaining_samples(self):
        return -1

    def has_more_samples(self):
        return True

    def next_sample(self, batch_size=1):

        """ next_sample

        An instance is generated based on the parameters passed. If noise
        is included the total number of attributes will be 24, if it's not
        included there will be 7 attributes.



        Parameters
        ----------
        batch_size: int
            The number of samples to return.

        Returns
        -------
        tuple or tuple list
            Return a tuple with the features matrix
            for the batch_size samples that were requested.

        """

        data = np.zeros([batch_size, self.n_cat_features + 1])

        for j in range(batch_size):
            selected = self.random_state.randint(10)

            for i in range(self._NUM_BASE_ATTRIBUTES):

                if (0.01 + self.random_state.rand()) <= self.noise_percentage:
                    data[j, i] = 1 if (self._ORIGINAL_INSTANCES[selected, i] == 0) else 0
                else:
                    data[j, i] = self._ORIGINAL_INSTANCES[selected, i]

            if self.has_noise():
                for i in range(self._NUM_BASE_ATTRIBUTES, self._TOTAL_ATTRIBUTES_INCLUDING_NOISE):
                    data[j, i] = self.random_state.randint(2)

        self.current_sample_x = data[:, :self.n_cat_features]
        return self.current_sample_x

    def restart(self):
        self.random_state = None
        self.prepare_for_use()

    def has_noise(self):
        return self.add_noise

    def get_name(self):
        return "Led Generator - {} target".format(self.n_targets)



    def get_info(self):
        return '  - n_num_features: ' + str(self.n_num_features) + \
               '  - add_noise: ' + str('True' if self.has_noise() else 'False') + \
               '  - noise_percentage: ' + str(self.noise_percentage) + \
               '  - random_state: ' + str(self.random_state)