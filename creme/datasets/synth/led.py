import numpy as np

from .. import base
from creme.utils.skmultiflow_utils import check_random_state


class LED(base.SyntheticDataset):
    """ LED stream generator.

    This data source originates from the CART book [^1]. An implementation
    in C was donated to the UCI [^2] machine learning repository by David Aha.
    The goal is to predict the digit displayed on a seven-segment LED display,
    where each attribute has a 10% chance of being inverted. It has an optimal
    Bayes classification rate of 74%. The particular configuration of the
    generator used for experiments (LED) produces 24 binary attributes,
    17 of which are irrelevant.

    Parameters
    ----------
    seed
        If int, `seed` is used to seed the random number generator;
        If RandomState instance, `seed` is the random number generator;
        If None, the random number generator is the `RandomState` instance used
        by `np.random`.

    noise_percentage
        The probability that noise will happen in the generation. At each
        new sample generated, a random number is generated, and if it is equal
        or less than the noise_percentage, the led value  will be switched

    has_noise
        Adds 17 non-relevant attributes to the stream.

    Notes
    -----
    An instance is generated based on the parameters passed. If `has_noise`
    is set then the total number of attributes will be 24, otherwise there will
    be 7 attributes.

    References
    ----------

    [^1]: Leo Breiman, Jerome Friedman, R. Olshen, and Charles J. Stone.
          Classification and Regression Trees. Wadsworth and Brooks,
          Monterey, CA,1984.

    [^2]: A. Asuncion and D. J. Newman. UCI Machine Learning Repository
          [http://www.ics.uci.edu/∼mlearn/mlrepository.html].
          University of California, Irvine, School of Information and
          Computer Sciences,2007.

    Examples
    --------
    >>> from creme import synth
    >>>
    >>> dataset = synth.LED(seed = 112, noise_percentage = 0.28, has_noise= False)
    >>>
    >>> for x, y in dataset.take(5):
    ...     print(x, y)
    {'x_0': 0, 'x_1': 1, 'x_2': 1, 'x_3': 1, 'x_4': 0, 'x_5': 0, 'x_6': 0} 4
    {'x_0': 0, 'x_1': 1, 'x_2': 0, 'x_3': 1, 'x_4': 0, 'x_5': 0, 'x_6': 0} 4
    {'x_0': 1, 'x_1': 0, 'x_2': 1, 'x_3': 1, 'x_4': 0, 'x_5': 0, 'x_6': 1} 3
    {'x_0': 0, 'x_1': 1, 'x_2': 1, 'x_3': 0, 'x_4': 0, 'x_5': 1, 'x_6': 1} 0
    {'x_0': 1, 'x_1': 1, 'x_2': 1, 'x_3': 1, 'x_4': 0, 'x_5': 1, 'x_6': 0} 4
    """
    _N_BASE_ATTRIBUTES = 7
    _N_ATTRIBUTES_INCLUDING_NOISE = 24
    _ORIGINAL_INSTANCES = np.array([[1, 1, 1, 0, 1, 1, 1],
                                    [0, 0, 1, 0, 0, 1, 0],
                                    [1, 0, 1, 1, 1, 0, 1],
                                    [1, 0, 1, 1, 0, 1, 1],
                                    [0, 1, 1, 1, 0, 1, 0],
                                    [1, 1, 0, 1, 0, 1, 1],
                                    [1, 1, 0, 1, 1, 1, 1],
                                    [1, 0, 1, 0, 0, 1, 0],
                                    [1, 1, 1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 0, 1, 1]], dtype=int)

    def __init__(self, seed: int or np.random.RandomState = None,
                 noise_percentage: float = 0.0, has_noise: bool = False):
        super().__init__(n_features=self._N_ATTRIBUTES_INCLUDING_NOISE if has_noise else
                         self._N_BASE_ATTRIBUTES, n_classes=10, n_outputs=1, task=base.MULTI_CLF)
        self.seed = seed
        self._rng = None  # This is the actual random_state object used internally
        if not (0.0 <= noise_percentage <= 1.0):
            raise ValueError(f"Invalid noise_percentage ({noise_percentage}). "
                             "Valid range is [0.0, 1.0]")
        self.noise_percentage = noise_percentage
        self.has_noise = has_noise

        self.n_cat_features = self.n_features

        self.feature_names = [f"x_{i}" for i in range(self.n_features)]
        self.target_values = [i for i in range(self.n_classes)]

    def __iter__(self):
        self._rng = check_random_state(self.seed)
        x = dict()

        while True:
            y = self._rng.randint(self.n_classes)

            for i, key in enumerate(self.feature_names[:self._N_BASE_ATTRIBUTES]):
                if (0.01 + self._rng.rand()) <= self.noise_percentage:
                    x[key] = 1 if (self._ORIGINAL_INSTANCES[y, i] == 0) else 0
                else:
                    x[key] = self._ORIGINAL_INSTANCES[y, i]

            if self.has_noise:
                for key in self.feature_names[self._N_BASE_ATTRIBUTES:]:
                    x[key] = self._rng.randint(2)

            yield x, y
