import random

from .. import base


class SEA(base.Dataset):
    """SEA generator.

    Each observation is composed of 3 features. Only the first two features are relevant. The
    target is binary, and is positive if the sum of the features exceeds a certain threshold. There
    are 4 thresholds to choose from. Concept drift can be introduced by switching the threshold
    anytime during the stream.

    Parameters:
        no (int): Determines the classification function to use. Possible choices are 0, 1, 2, 3.
        noise (float): Determines the amount of observations for which the target sign will be
            flipped.

    Example:

        >>> from creme import datasets

        >>> gen = datasets.gen.SEA(no=0, seed=42)

        >>> for x, y in gen.take(5):
        ...     print(x, y)
        {0: 6.394267984578837, 1: 0.25010755222666936, 2: 2.7502931836911926} False
        {0: 2.2321073814882277, 1: 7.364712141640124, 2: 6.766994874229113} True
        {0: 8.921795677048454, 1: 0.8693883262941615, 2: 4.2192181968527045} True
        {0: 0.29797219438070344, 1: 2.1863797480360336, 2: 5.053552881033624} False
        {0: 0.26535969683863625, 1: 1.988376506866485, 2: 6.498844377795232} False

    References:
        1. `A Streaming Ensemble Algorithm (SEA) for Large-Scale Classification <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.482.3991&rep=rep1&type=pdf>`_

    """

    def __init__(self, no, noise=0., seed=None):

        super().__init__(n_features=3, category=base.BINARY_CLF)

        if no not in (0, 1, 2, 3):
            raise ValueError('no must be one of (0, 1, 2, 3)')

        self.no = no
        self.noise = noise
        self.seed = seed
        self._threshold = {
            0: 8,
            1: 9,
            2: 7,
            3: 9.5
        }[no]
        self._rng = random.Random(seed)

    def __iter__(self):

        while True:

            x = {i: self._rng.uniform(0, 10) for i in range(3)}
            y = x[0] + x[1] > self._threshold

            if self.noise and self._rng.random() < self.noise:
                y = not y

            yield x, y
