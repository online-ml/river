import warnings

import numpy as np
from skmultiflow.data.base_stream import Stream
from skmultiflow.utils import check_random_state
from skmultiflow.data import AGRAWALGenerator


class ConceptDriftStream(Stream):
    """ Generates a stream with concept drift.

    A stream generator that adds concept drift or change by joining several streams.
    This is done by building a weighted combination of two pure distributions that
    characterizes the target concepts before and after the change.

    The sigmoid function is an elegant and practical solution to define the probability that each
    new instance of the stream belongs to the new concept after the drift. The sigmoid function
    introduces a gradual, smooth transition whose duration is controlled with two parameters:

    - :math:`p`, the position of the change.
    - :math:`w`, the width of the transition.

    The sigmoid function at sample `t` is :math:`f(t) = 1/(1+e^{-4(t-p)/w})`.

    Parameters
    ----------
    stream: Stream (default= AGRAWALGenerator(random_state=112))
        Original stream concept

    drift_stream: Stream (default= AGRAWALGenerator(random_state=112, classification_function=2))
        Drift stream concept

    random_state: int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    alpha: float (optional, default: None)
        Angle of change to estimate the width of concept drift change.
        If set will override the width parameter. Valid values are in the range (0.0, 90.0].
        If alpha is None, this parameter will be ignored.

    position: int (default: 5000)
        Central position of concept drift change.

    width: int (Default: 1000)
        Width of concept drift change.

    Notes
    -----
    An optional way to estimate the width of the transition :math:`w` is based on
    the angle :math:`\\alpha`: :math:`w = 1/ tan(\\alpha)`. Since width corresponds to
    the number of samples for the transition, the width is round-down to the nearest
    smaller integer. Notice that larger values of :math:`\\alpha` result in smaller widths.
    For :math:`\\alpha>45.0`, the width is smaller than 1 so values are round-up to 1 to avoid
    division by zero errors.

    """

    def __init__(self, stream=AGRAWALGenerator(random_state=112),
                 drift_stream=AGRAWALGenerator(random_state=112, classification_function=2),
                 position=5000,
                 width=1000,
                 random_state=None,
                 alpha=None):
        super(ConceptDriftStream, self).__init__()

        self.n_samples = stream.n_samples
        self.n_targets = stream.n_targets
        self.n_features = stream.n_features
        self.n_num_features = stream.n_num_features
        self.n_cat_features = stream.n_cat_features
        self.n_classes = stream.n_classes
        self.cat_features_idx = stream.cat_features_idx
        self.feature_names = stream.feature_names
        self.target_names = stream.target_names
        self.target_values = stream.target_values
        self.n_targets = stream.n_targets
        self.name = 'Drifting' + stream.name

        self.random_state = random_state
        self._random_state = None   # This is the actual random_state object used internally
        self.alpha = alpha
        if self.alpha == 0:
            warnings.warn("Default value for 'alpha' has changed from 0 to None. 'alpha=0' will "
                          "throw an error from v0.7.0", category=FutureWarning)
            self.alpha = None
        if self.alpha is not None:
            if 0 < self.alpha <= 90.0:
                w = int(1 / np.tan(self.alpha * np.pi / 180))
                self.width = w if w > 0 else 1
            else:
                raise ValueError('Invalid alpha value: {}'.format(alpha))
        else:
            self.width = width
        self.position = position
        self.stream = stream
        self.drift_stream = drift_stream

        self._prepare_for_use()

    def _prepare_for_use(self):
        self._random_state = check_random_state(self.random_state)

    def n_remaining_samples(self):
        """ Returns the estimated number of remaining samples.

        Returns
        -------
        int
            Remaining number of samples. -1 if infinite (e.g. generator)
        """
        n_samples = self.stream.n_remaining_samples() + self.drift_stream.n_remaining_samples()
        if n_samples < 0:
            n_samples = -1
        return n_samples

    def has_more_samples(self):
        """ Checks if stream has more samples.

        Returns
        -------
        Boolean
            True if stream has more samples.
        """
        return self.stream.has_more_samples() and self.drift_stream.has_more_samples()

    def is_restartable(self):
        """ Determine if the stream is restartable.

         Returns
         -------
         Boolean
            True if stream is restartable.
         """
        return self.stream.is_restartable() and self.drift_stream.is_restartable()

    def next_sample(self, batch_size=1):
        """ Returns next sample from the stream.

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
        self.current_sample_x = np.zeros((batch_size, self.n_features))
        self.current_sample_y = np.zeros((batch_size, self.n_targets))

        for j in range(batch_size):
            self.sample_idx += 1
            x = -4.0 * float(self.sample_idx - self.position) / float(self.width)
            probability_drift = 1.0 / (1.0 + np.exp(x))
            if self._random_state.rand() > probability_drift:
                X, y = self.stream.next_sample()
            else:
                X, y = self.drift_stream.next_sample()
            self.current_sample_x[j, :] = X
            self.current_sample_y[j, :] = y

        return self.current_sample_x, self.current_sample_y.flatten()

    def restart(self):
        self._random_state = check_random_state(self.random_state)
        self.sample_idx = 0
        self.stream.restart()
        self.drift_stream.restart()
