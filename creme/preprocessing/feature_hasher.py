import collections

from sklearn import utils

from .. import base


__all__ = ['FeatureHasher']


class FeatureHasher(base.Transformer):
    """Implements the hashing trick.

    Each pair of (name, value) features is hashed into a random integer. A module operator is then
    used to make sure the hash is in a certain range. We use the Murmurhash implementation from
    scikit-learn.

    Parameters:
        n_features (int): The number by which each hash will be moduloed by.
        alternate_sign (bool): Whether or not half of the hashes will be negated.
        encoding (str): The string encoding used by `sklearn.utils.murmurhash3_32`.
        random_state (int, ``numpy.random.RandomState`` instance or None): If int, ``random_state``
            is the seed used by the random number generator; if ``RandomState`` instance,
            ``random_state`` is the random number generator; if ``None``, the random number
            generator is the ``RandomState`` instance used by `numpy.random`.

    Arguments:
        seed (int): Seed used by `sklearn.utils.murmurhash3_32`.

    Example:

        ::

            >>> import creme
            >>> hasher = creme.preprocessing.FeatureHasher(n_features=10, random_state=0)
            >>> X = [{'dog': 1, 'cat': 2, 'elephant': 4}, {'dog': 2, 'run': 5}]
            >>> for x in X:
            ...     print(hasher.transform_one(x))
            defaultdict(<class 'int'>, {2: -1, 1: 2, 8: 4})
            defaultdict(<class 'int'>, {2: -2, 4: 5})

    References:

        1. `Feature vectorization using hashing trick <https://www.wikiwand.com/en/Feature_hashing#/Feature_vectorization_using_hashing_trick>`_

    """

    def __init__(self, n_features=1048576, alternate_sign=True, encoding='utf-8',
                 random_state=None):
        self.n_features = n_features
        self.alternate_sign = alternate_sign
        self.encoding = encoding
        self.random_state = utils.check_random_state(random_state)
        self.seed = self.random_state.randint(0, 2 ** 32 - 1)

    def _hash_str(self, s):
        return utils.murmurhash3_32(bytes(s, self.encoding), self.seed)

    def transform_one(self, x):

        x_hashed = collections.defaultdict(int)

        for feature, value in x.items():

            if isinstance(value, str):
                feature = f'{feature}={value}'
                value = 1

            if value == 0:
                continue

            h = self._hash_str(feature)
            i = h % self.n_features

            if self.alternate_sign:
                value *= (h >= 0) * 2 - 1

            x_hashed[i] += value

        return x_hashed
