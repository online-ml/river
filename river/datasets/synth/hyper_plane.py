import numpy as np

from river.utils.skmultiflow_utils import check_random_state

from .. import base


class Hyperplane(base.SyntheticDataset):
    r"""Hyperplane stream generator.

    Generates a problem of prediction class of a rotation hyperplane. It was
    used as testbed for CVFDT and VFDT in [^1].

    A hyperplane in d-dimensional space is the set of points $x$ that satisfy

    $$\sum^{d}_{i=1} w_i x_i = w_0 = \sum^{d}_{i=1} w_i$$

    where $x_i$ is the i-th coordinate of $x$.

    - Examples for which $\sum^{d}_{i=1} w_i x_i > w_0$, are labeled positive.

    - Examples for which $\sum^{d}_{i=1} w_i x_i \leq w_0$, are labeled negative.

    Hyperplanes are useful for simulating time-changing concepts because we
    can change the orientation and position of the hyperplane in a smooth
    manner by changing the relative size of the weights. We introduce change
    to this dataset by adding drift to each weighted feature
    $w_i = w_i + d \sigma$, where $\sigma$ is the probability that the
    direction of change is reversed and $d$ is the change applied to each
    example.

    Parameters
    ----------
    seed
        If int, `seed` is used to seed the random number generator;
        If RandomState instance, `seed` is the random number generator;
        If None, the random number generator is the `RandomState` instance used
        by `np.random`.
    n_features
        The number of attributes to generate. Higher than 2.
    n_drift_features
        The number of attributes with drift. Higher than 2.
    mag_change
        Magnitude of the change for every example. From 0.0 to 1.0.
    noise_percentage
        Percentage of noise to add to the data. From 0.0 to 1.0.
    sigma
        Probability that the direction of change is reversed. From 0.0 to 1.0.

    Examples
    --------

    >>> from river import synth

    >>> dataset = synth.Hyperplane(seed=42, n_features=2)

    >>> for x, y in dataset.take(5):
    ...     print(x, y)
    {0: 0.7319, 1: 0.5986} 1
    {0: 0.8661, 1: 0.6011} 1
    {0: 0.8324, 1: 0.2123} 0
    {0: 0.5247, 1: 0.4319} 0
    {0: 0.2921, 1: 0.3663} 0

    Notes
    -----
    The sample generation works as follows: The features are generated
    with the random number generator, initialized with the seed passed by
    the user. Then the classification function decides, as a function of
    the sum of the weighted features and the sum of the weights, whether
    the instance belongs to class 0 or class 1. The last step is to add
    noise and generate drift.

    References
    ----------
    [^1]: G. Hulten, L. Spencer, and P. Domingos. Mining time-changing data streams.
          In KDD’01, pages 97–106, San Francisco, CA, 2001. ACM Press.

    """

    def __init__(
        self,
        seed: int or np.random.RandomState = None,
        n_features: int = 10,
        n_drift_features: int = 2,
        mag_change: float = 0.0,
        noise_percentage: float = 0.05,
        sigma: float = 0.1,
    ):
        super().__init__(
            n_features=n_features, n_classes=2, n_outputs=1, task=base.BINARY_CLF
        )

        self.seed = seed
        self.n_drift_features = n_drift_features
        if not (0.0 <= mag_change <= 1.0):
            raise ValueError(
                f"Invalid mag_change ({mag_change}). " "Valid range is [0.0, 1.0]"
            )
        self.mag_change = mag_change
        if not (0.0 <= sigma <= 1.0):
            raise ValueError(
                f"Invalid sigma_percentage ({sigma}). " "Valid range is [0.0, 1.0]"
            )
        self.sigma = sigma
        if not (0.0 <= noise_percentage <= 1.0):
            raise ValueError(
                f"Invalid noise_percentage ({noise_percentage}). "
                "Valid range is [0.0, 1.0]"
            )
        self.noise_percentage = noise_percentage
        self.target_values = [0, 1]

    def __iter__(self):
        self._rng = check_random_state(self.seed)
        self._weights = self._rng.rand(self.n_features)
        self._change_direction = np.zeros(self.n_features)
        self._change_direction[: self.n_drift_features] = 1

        while True:
            x = dict()

            sum_weights = np.sum(self._weights)
            sum_value = 0
            for i in range(self.n_features):
                x[i] = self._rng.rand()
                sum_value += self._weights[i] * x[i]

            y = 1 if sum_value >= sum_weights * 0.5 else 0

            if 0.01 + self._rng.rand() <= self.noise_percentage:
                y = 1 if (y == 0) else 0

            self._generate_drift()

            yield x, y

    def _generate_drift(self):
        for i in range(self.n_drift_features):
            self._weights[i] += self._change_direction[i] * self.mag_change
            if (0.01 + self._rng.rand()) <= self.sigma:
                self._change_direction[i] *= -1
