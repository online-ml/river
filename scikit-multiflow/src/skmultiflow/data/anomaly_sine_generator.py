from skmultiflow.data.base_stream import Stream
from skmultiflow.utils import check_random_state

import numpy as np


class AnomalySineGenerator(Stream):
    """
    Simulate a stream with anomalies in sine waves

    Parameters
    ----------
    n_samples: int, optional (default=10000)
        Number of samples
    n_anomalies: int, optional (default=2500)
        Number of anomalies. Can't be larger than n_samples.
    contextual: bool, optional (default=False)
        If True, will add contextual anomalies
    n_contextual: int, optional (default=2500)
        Number of contextual anomalies. Can't be larger than n_samples.
    shift: int, optional (default=4)
        Shift applied when retrieving contextual anomalies
    noise: float, optional (default=0.5)
        Amount of noise
    replace: bool, optional (default=True)
        If True, anomalies are randomly sampled with replacement
    random_state: int, RandomState instance or None, optional (default=None)
    If int, random_state is the seed used by the random number generator;
    If RandomState instance, random_state is the random number generator;
    If None, the random number generator is the RandomState instance used
    by `np.random`.

    Notes
    -----
    The data generated corresponds to sine (attribute 1) and cosine
    (attribute 2) functions. Anomalies are induced by replacing values
    from attribute 2 with values from a sine function different to the one
    used in attribute 1. The ``contextual`` flag can be used to introduce
    contextual anomalies which are values in the normal global range,
    but abnormal compared to the seasonal pattern. Contextual attributes
    are introduced by replacing values in attribute 2 with values from
    attribute 1.

    """

    def __init__(self, n_samples=10000, n_anomalies=2500, contextual=False,
                 n_contextual=2500, shift=4, noise=0.5, replace=True, random_state=None):
        super().__init__()
        self.n_samples = n_samples
        if n_anomalies > self.n_samples:
            raise ValueError("n_anomalies ({}) can't be larger "
                             "than n_samples ({})".format(n_anomalies, self.n_samples))
        self.n_anomalies = n_anomalies
        self.contextual = contextual
        self.n_contextual = n_contextual
        if contextual and n_contextual > self.n_samples:
            raise ValueError("n_contextual ({}) can't be larger "
                             "than n_samples ({})".format(n_contextual, self.n_samples))
        self.shift = abs(shift)
        self.noise = noise
        self.replace = replace
        self.random_state = random_state
        self._random_state = None  # This is the actual random_state object used internally
        self.name = 'Anomaly Sine Generator'
        self.restart()

        # Stream attributes
        self.n_features = 2
        self.n_targets = 1
        self.n_num_features = 2
        self.target_names = ["anomaly"]
        self.feature_names = ["att_idx_" + str(i) for i in range(2)]
        self.target_values = [0, 1]

        self._prepare_for_use()

    def _prepare_for_use(self):
        self._random_state = check_random_state(self.random_state)
        self.y = np.zeros(self.n_samples)
        self.X = np.column_stack(
            [np.sin(np.arange(self.n_samples) / 4.)
             + self._random_state.randn(self.n_samples) * self.noise,
             np.cos(np.arange(self.n_samples) / 4.)
             + self._random_state.randn(self.n_samples) * self.noise]
        )

        if self.contextual:
            # contextual anomaly indices
            contextual_anomalies = self._random_state.choice(self.n_samples - self.shift,
                                                             self.n_contextual,
                                                             replace=self.replace)
            # set contextual anomalies
            contextual_idx = contextual_anomalies + self.shift
            contextual_idx[contextual_idx >= self.n_samples] -= self.n_samples
            self.X[contextual_idx, 1] = self.X[contextual_anomalies, 0]

        # Anomaly indices
        anomalies_idx = self._random_state.choice(self.n_samples, self.n_anomalies,
                                                  replace=self.replace)
        self.X[anomalies_idx, 1] = np.sin(self._random_state.choice(self.n_anomalies,
                                                                    replace=self.replace)) \
            + self._random_state.randn(self.n_anomalies) * self.noise + 2.
        # Mark sample as anomalous
        self.y[anomalies_idx] = 1

    def next_sample(self, batch_size=1):
        """
        Get the next sample from the stream.

        Parameters
        ----------
        batch_size: int (optional, default=1)
            The number of samples to return.

        Returns
        -------
        tuple of arrays
            Return a tuple with the features X and the target y for
            the batch_size samples that are requested.

        """
        if self.n_remaining_samples() < batch_size:
            batch_size = self.n_remaining_samples()

        if batch_size > 0:
            self.sample_idx += batch_size
            self.current_sample_x = self.X[self.sample_idx - batch_size:self.sample_idx, :]
            self.current_sample_y = self.y[self.sample_idx - batch_size:self.sample_idx].flatten()
        else:
            self.current_sample_x = None
            self.current_sample_y = None

        return self.current_sample_x, self.current_sample_y

    def n_remaining_samples(self):
        """
        Returns
        -------
        int
            Number of samples remaining.
        """
        return self.n_samples - self.sample_idx

    def get_data_info(self):
        """ Retrieves minimum information from the stream

        Used by evaluator methods to id the stream.

        The default format is: 'Stream name - n_targets, n_classes, n_features'.

        Returns
        -------
        string
            Stream data information

        """
        return self.name + " - {} target(s), {} features".format(self.n_targets, self.n_features)

    def restart(self):
        """
        Restart the stream to the initial state.

        """
        # Note: No need to regenerate the data, just reset the idx
        self.sample_idx = 0
        self.current_sample_x = None
        self.current_sample_y = None
