from __future__ import annotations

import random

from river import datasets


class LED(datasets.base.SyntheticDataset):
    """LED stream generator.

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
        Random seed for reproducibility.
    noise_percentage
        The probability that noise will happen in the generation. At each
        new sample generated, a random number is generated, and if it is equal
        or less than the noise_percentage, the led value  will be switched
    irrelevant_features
        Adds 17 non-relevant attributes to the stream.

    Examples
    --------
    >>> from river.datasets import synth

    >>> dataset = synth.LED(seed = 112, noise_percentage = 0.28, irrelevant_features= False)

    >>> for x, y in dataset.take(5):
    ...     print(x, y)
    {0: 1, 1: 0, 2: 1, 3: 0, 4: 0, 5: 1, 6: 0} 7
    {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 0} 8
    {0: 1, 1: 1, 2: 1, 3: 1, 4: 0, 5: 1, 6: 0} 9
    {0: 0, 1: 0, 2: 1, 3: 0, 4: 0, 5: 1, 6: 0} 1
    {0: 0, 1: 1, 2: 1, 3: 0, 4: 0, 5: 0, 6: 0} 1

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
          [http://www.ics.uci.edu/~mlearn/mlrepository.html].
          University of California, Irvine, School of Information and
          Computer Sciences,2007.

    """

    _N_RELEVANT_FEATURES = 7
    _N_FEATURES_INCLUDING_NOISE = 24
    _ORIGINAL_INSTANCES = [
        [1, 1, 1, 0, 1, 1, 1],
        [0, 0, 1, 0, 0, 1, 0],
        [1, 0, 1, 1, 1, 0, 1],
        [1, 0, 1, 1, 0, 1, 1],
        [0, 1, 1, 1, 0, 1, 0],
        [1, 1, 0, 1, 0, 1, 1],
        [1, 1, 0, 1, 1, 1, 1],
        [1, 0, 1, 0, 0, 1, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 0, 1, 1],
    ]

    def __init__(
        self,
        seed: int | None = None,
        noise_percentage: float = 0.0,
        irrelevant_features: bool = False,
    ):
        super().__init__(
            n_features=self._N_FEATURES_INCLUDING_NOISE
            if irrelevant_features
            else self._N_RELEVANT_FEATURES,
            n_classes=10,
            n_outputs=1,
            task=datasets.base.MULTI_CLF,
        )
        self.seed = seed
        self._rng = None
        if not (0.0 <= noise_percentage <= 1.0):
            raise ValueError(
                f"Invalid noise_percentage ({noise_percentage}). " "Valid range is [0.0, 1.0]"
            )
        self.noise_percentage = noise_percentage
        self.irrelevant_features = irrelevant_features
        self.n_cat_features = self.n_features
        self.target_values = [i for i in range(self.n_classes)]

    def __iter__(self):
        self._rng = random.Random(self.seed)

        while True:
            x = dict()
            y = self._rng.randint(0, self.n_classes - 1)

            for i in range(self._N_RELEVANT_FEATURES):
                if (0.01 + self._rng.random()) <= self.noise_percentage:
                    x[i] = int(self._ORIGINAL_INSTANCES[y][i] == 0)
                else:
                    x[i] = self._ORIGINAL_INSTANCES[y][i]

            if self.irrelevant_features:
                for i in range(self._N_RELEVANT_FEATURES, self._N_FEATURES_INCLUDING_NOISE):
                    x[i] = self._rng.choice([0, 1])

            yield x, y


class LEDDrift(LED):
    """LED stream generator with concept drift.

    This class is an extension of the `LED` generator whose purpose is to add
    concept drift to the stream.

    Parameters
    ----------
    seed
        Random seed for reproducibility.
    noise_percentage
        The probability that noise will happen in the generation. At each
        new sample generated, a random number is generated, and if it is equal
        or less than the noise_percentage, the led value  will be switched
    irrelevant_features
        Adds 17 non-relevant attributes to the stream.
    n_drift_features
        The number of attributes that have drift.

    Examples
    --------
    >>> from river.datasets import synth

    >>> dataset = synth.LEDDrift(seed = 112, noise_percentage = 0.28,
    ...                          irrelevant_features= True, n_drift_features=4)

    >>> for x, y in dataset.take(5):
    ...     print(list(x.values()), y)
    [1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1] 7
    [1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0] 6
    [0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1] 1
    [1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1] 6
    [1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0] 7

    Notes
    -----
    An instance is generated based on the parameters passed. If `has_noise`
    is set then the total number of attributes will be 24, otherwise there will
    be 7 attributes.

    """

    _N_IRRELEVANT_ATTRIBUTES = 17

    def __init__(
        self,
        seed: int | None = None,
        noise_percentage: float = 0.0,
        irrelevant_features: bool = False,
        n_drift_features: int = 0,
    ):
        super().__init__(
            seed=seed,
            noise_percentage=noise_percentage,
            irrelevant_features=irrelevant_features,
        )
        self.n_drift_features = n_drift_features

    def __iter__(self):
        self._rng = random.Random(self.seed)
        self._attr_idx = list(range(self._N_FEATURES_INCLUDING_NOISE))

        # Change attributes
        if self.irrelevant_features and self.n_drift_features > 0:
            random_int = self._rng.randint(0, 6)
            offset = self._rng.randint(0, self._N_IRRELEVANT_ATTRIBUTES - 1)
            for i in range(self.n_drift_features):
                value_1 = (i + random_int) % 7
                value_2 = 7 + (i + offset) % self._N_IRRELEVANT_ATTRIBUTES
                self._attr_idx[value_1] = value_2
                self._attr_idx[value_2] = value_1

        while True:
            x = {i: -1 for i in range(self.n_features)}  # Initialize to keep order in dictionary
            y = self._rng.randint(0, self.n_classes - 1)

            for i in range(self._N_RELEVANT_FEATURES):
                if (0.01 + self._rng.random()) <= self.noise_percentage:
                    x[self._attr_idx[i]] = int(self._ORIGINAL_INSTANCES[y][i] == 0)
                else:
                    x[self._attr_idx[i]] = self._ORIGINAL_INSTANCES[y][i]
            if self.irrelevant_features:
                for i in range(self._N_RELEVANT_FEATURES, self._N_FEATURES_INCLUDING_NOISE):
                    x[self._attr_idx[i]] = self._rng.choice([0, 1])

            yield x, y
