import numpy as np
from skmultiflow.data.base_stream import Stream
from skmultiflow.core.utils.validation import check_random_state
from skmultiflow.data.generators.agrawal_generator import AGRAWALGenerator


class ConceptDriftStream(Stream):
    """ ConceptDriftStream
    A stream generator that adds concept drift or change by joining several streams.
    This is done by building a weighted combination of two pure distributions that
    characterizes the target concepts before and after the change.
    MOA uses the sigmoid function as an elegant and practical solution to define
    the probability that each new instance of the stream belongs to the new concept after the drift.
    The sigmoid function introduces a gradual, smooth transition whose duration is controlled with
    two parameters: p, the position where the change occurs, and the length w of the transition
    :math:`f(t) = 1/(1+\e^{-4*(t-p)/w})`

    Parameters
    ----------
    stream_option: generator (Default= AGRAWALGenerator(random_state=112))
        stream generator

    drift_stream_option: generator (Default= AGRAWALGenerator(random_state=112,classification_function=2))
        stream generator that adds drift

    random_state: int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    alpha_option: float (Default: 0.0)
        Angle alpha of change grade
        Values go from 0.0 to 90.0

    position_option: int (Default: 0)
        Central position of concept drift change

    position_option: int (Default: 1000)
        Width of concept drift change
    """

    def __init__(self, stream_option=AGRAWALGenerator(random_state=112),
                 drift_stream_option=AGRAWALGenerator(random_state=112, classification_function=2),
                 random_state=None, alpha_option=0.0,
                 position_option=0, width_option=1000):

        super().__init__()

        self.n_samples = stream_option.n_samples
        self.n_targets = stream_option.n_targets
        self.n_features = stream_option.n_features
        self.n_num_features = stream_option.n_num_features
        self.n_cat_features = stream_option.n_cat_features
        self.n_classes = stream_option.n_classes
        self.cat_features_idx = stream_option.cat_features_idx
        self.feature_names = stream_option.feature_names
        self.target_names = stream_option.target_values
        self.target_values = stream_option.target_values
        self.name = stream_option.name

        self._original_random_state = random_state
        self.random_state = None
        self.alpha_option = alpha_option
        self.position_option = position_option
        self.width_option = width_option
        self._input_stream = stream_option
        self._drift_stream = drift_stream_option
        self.n_targets = stream_option.n_targets

    def prepare_for_use(self):
        self.random_state = check_random_state(self._original_random_state)
        self.sample_idx = 0
        self._input_stream.prepare_for_use()
        self._drift_stream.prepare_for_use()
        if self.alpha_option != 0.0:
            self.width_option = int(1 / np.tan(self.alpha_option * np.pi / 180))

    def n_remaining_samples(self):
        return self._input_stream.n_remaining_samples() + self._drift_stream.n_remaining_samples()

    def has_more_samples(self):
        return self._input_stream.has_more_samples() and self._drift_stream.has_more_samples()

    def is_restartable(self):
        return self._input_stream.is_restartable()

    def next_sample(self, batch_size=1):

        """ next_sample
        An instance is generated based on the parameters passed.

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
        self.current_sample_x = []
        self.current_sample_y = []

        for j in range(batch_size):
            self.sample_idx += 1
            x = -4.0 * float(self.sample_idx - self.position_option) / float(self.width_option)
            probability_drift = 1.0 / (1.0 + np.exp(x))
            if self.random_state.rand() > probability_drift:
                if self.current_sample_x == []:
                    self.current_sample_x, self.current_sample_y = self._input_stream.next_sample()
                else:
                    X, y = self._input_stream.next_sample()
                    self.current_sample_x = np.append(self.current_sample_x, X, axis=0)
                    self.current_sample_y = np.append(self.current_sample_y, y, axis=0)
            else:
                if self.current_sample_x == []:
                    self.current_sample_x, self.current_sample_y = self._drift_stream.next_sample()
                else:
                    X, y = self._drift_stream.next_sample()
                    self.current_sample_x = np.append(self.current_sample_x, X, axis=0)
                    self.current_sample_y = np.append(self.current_sample_y, y, axis=0)

        return self.current_sample_x, self.current_sample_y

    def restart(self):
        self.prepare_for_use()

    def get_info(self):
        pass
