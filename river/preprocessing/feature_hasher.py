from __future__ import annotations

import collections
import hashlib

import numpy as np

from river import base

__all__ = ["FeatureHasher"]


class FeatureHasher(base.Transformer):
    """Implements the hashing trick.

    Each pair of (name, value) features is hashed into a random integer. A module operator is then
    used to make sure the hash is in a certain range. We use the Murmurhash implementation from
    scikit-learn.

    Parameters
    ----------
    n_features
        The number by which each hash will be moduloed by.
    alternate_sign
        Whether or not half of the hashes will be negated.
    seed
        Set the seed to produce identical results.

    Examples
    --------

    >>> import river

    >>> hasher = river.preprocessing.FeatureHasher(n_features=10, seed=42)

    >>> X = [
    ...     {'dog': 1, 'cat': 2, 'elephant': 4},
    ...     {'dog': 2, 'run': 5}
    ... ]
    >>> for x in X:
    ...     print(hasher.transform_one(x))
    Counter({1: 4, 9: 2, 8: 1})
    Counter({4: 5, 8: 2})

    References
    ----------
    [^1]: [Wikipedia article on feature vectorization using the hashing trick](https://www.wikiwand.com/en/Feature_hashing#/Feature_vectorization_using_hashing_trick)

    """

    def __init__(self, n_features=1048576, seed: int | None = None):
        self.n_features = n_features
        self.seed = seed
        self._salt = np.random.RandomState(seed).bytes(hashlib.blake2s.SALT_SIZE)

    def _hash(self, x):
        hexa = hashlib.blake2s(bytes(x, encoding="utf8"), salt=self._salt).hexdigest()
        return int(hexa, 16)

    def transform_one(self, x):
        x_hashed = collections.Counter()

        for feature, value in x.items():
            if isinstance(value, str):
                feature = f"{feature}={value}"
                value = 1

            i = self._hash(feature) % self.n_features
            x_hashed[i] += value

        return x_hashed
