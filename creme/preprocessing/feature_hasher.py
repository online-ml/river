import collections
import random

import mmh3

from .. import base


__all__ = ['FeatureHasher']


class FeatureHasher(base.Transformer):
    """Implements the hashing trick.

    Each pair of (name, value) features is hashed into a random integer. A module operator is then
    used to make sure the hash is in a certain range. We use the Murmurhash implementation from
    scikit-learn.

    Parameters:
        n_features: The number by which each hash will be moduloed by.
        alternate_sign: Whether or not half of the hashes will be negated.
        seed: Set the seed to produce identical results.

    Example:

        >>> import creme

        >>> hasher = creme.preprocessing.FeatureHasher(n_features=10, seed=0)

        >>> X = [
        ...     {'dog': 1, 'cat': 2, 'elephant': 4},
        ...     {'dog': 2, 'run': 5}
        ... ]
        >>> for x in X:
        ...     print(hasher.transform_one(x))
        defaultdict(<class 'int'>, {0: 1, 3: -2, 4: 4})
        defaultdict(<class 'int'>, {0: -3})

    References:
        1. [Wikipedia article on feature vectorization using the hashing trick](https://www.wikiwand.com/en/Feature_hashing#/Feature_vectorization_using_hashing_trick)

    """

    def __init__(self, n_features=1048576, alternate_sign=True, seed: int = None):
        self.n_features = n_features
        self.alternate_sign = alternate_sign
        self.seed = seed
        self._hash_seed = random.Random(seed).randint(0, 2 ** 10)

    def transform_one(self, x):

        x_hashed = collections.defaultdict(int)

        for feature, value in x.items():

            if isinstance(value, str):
                feature = f'{feature}={value}'
                value = 1

            if value == 0:
                continue

            h = mmh3.hash(feature, seed=self._hash_seed)
            i = h % self.n_features

            if self.alternate_sign:
                value *= (h >= 0) * 2 - 1

            x_hashed[i] += value

        return x_hashed
