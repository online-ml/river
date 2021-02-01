import itertools

import numpy as np

from river.utils.skmultiflow_utils import check_random_state

from .. import base


class AnomalySine(base.SyntheticDataset):
    """Simulate a stream with anomalies in sine waves

    The data generated corresponds to sine (`attribute 1`) and cosine
    (`attribute 2`) functions. Anomalies are induced by replacing values
    from `attribute 2` with values from a sine function different to the one
    used in `attribute 1`. The `contextual` flag can be used to introduce
    contextual anomalies which are values in the normal global range,
    but abnormal compared to the seasonal pattern. Contextual attributes
    are introduced by replacing values in `attribute 2` with values from
    `attribute 1`.

    Parameters
    ----------
    n_samples
        Number of samples
    n_anomalies
        Number of anomalies. Can't be larger than `n_samples`.
    contextual
        If True, will add contextual anomalies
    n_contextual
        Number of contextual anomalies. Can't be larger than `n_samples`.
    shift
        Shift in number of samples applied when retrieving contextual anomalies
    noise
        Amount of noise
    replace
        If True, anomalies are randomly sampled with replacement
    seed
        If int, `seed` is used to seed the random number generator;
        If RandomState instance, `seed` is the random number generator;
        If None, the random number generator is the `RandomState` instance used
        by `np.random`.

    Examples
    --------

    >>> from river import synth

    >>> dataset = synth.AnomalySine(seed=12345,
    ...                             n_samples=100,
    ...                             n_anomalies=25,
    ...                             contextual=True,
    ...                             n_contextual=10)

    >>> for x, y in dataset.take(5):
    ...     print(x, y)
    {'sine': -0.1023, 'cosine': 0.2171} 0.0
    {'sine': 0.4868, 'cosine': 0.6876} 0.0
    {'sine': 0.2197, 'cosine': 0.8612} 0.0
    {'sine': 0.4037, 'cosine': 0.2671} 0.0
    {'sine': 1.8243, 'cosine': 1.8268} 1.0

    """

    def __init__(
        self,
        n_samples: int = 10000,
        n_anomalies: int = 2500,
        contextual: bool = False,
        n_contextual: int = 2500,
        shift: int = 4,
        noise: float = 0.5,
        replace: bool = True,
        seed: int or np.random.RandomState = None,
    ):
        super().__init__(
            n_features=2,
            n_classes=1,
            n_outputs=1,
            n_samples=n_samples,
            task=base.BINARY_CLF,
        )
        if n_anomalies > self.n_samples:
            raise ValueError(
                f"n_anomalies ({n_anomalies}) can't be larger "
                f"than n_samples ({self.n_samples})"
            )
        self.n_anomalies = n_anomalies
        self.contextual = contextual
        if contextual and n_contextual > self.n_samples:
            raise ValueError(
                f"n_contextual ({n_contextual}) can't be larger "
                f"than n_samples ({self.n_samples})"
            )
        self.n_contextual = n_contextual
        self.shift = abs(shift)
        self.noise = noise
        self.replace = replace
        self.seed = seed

        # Stream attributes
        self.n_num_features = 2

    def _generate_data(self):
        # Generate anomaly data arrays
        self._random_state = check_random_state(self.seed)
        self.y = np.zeros(self.n_samples)
        self.X = np.column_stack(
            [
                np.sin(np.arange(self.n_samples) / 4.0)
                + self._random_state.randn(self.n_samples) * self.noise,
                np.cos(np.arange(self.n_samples) / 4.0)
                + self._random_state.randn(self.n_samples) * self.noise,
            ]
        )

        if self.contextual:
            # contextual anomaly indices
            contextual_anomalies = self._random_state.choice(
                self.n_samples - self.shift, self.n_contextual, replace=self.replace
            )
            # set contextual anomalies
            contextual_idx = contextual_anomalies + self.shift
            contextual_idx[contextual_idx >= self.n_samples] -= self.n_samples
            self.X[contextual_idx, 1] = self.X[contextual_anomalies, 0]

        # Anomaly indices
        anomalies_idx = self._random_state.choice(
            self.n_samples, self.n_anomalies, replace=self.replace
        )
        self.X[anomalies_idx, 1] = (
            np.sin(self._random_state.choice(self.n_anomalies, replace=self.replace))
            + self._random_state.randn(self.n_anomalies) * self.noise
            + 2.0
        )
        # Mark sample as anomalous
        self.y[anomalies_idx] = 1

    def __iter__(self):

        self._generate_data()

        for xi, yi in itertools.zip_longest(
            self.X, self.y if hasattr(self.y, "__iter__") else []
        ):
            yield dict(zip(["sine", "cosine"], xi)), yi
