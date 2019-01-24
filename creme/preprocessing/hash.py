import collections

from sklearn import utils

from .. import base


__all__ = ['FeatureHasher']


class FeatureHasher(base.Transformer):
    """
    Example
    -------

        #!python
        >>> import creme
        >>> hasher = creme.preprocessing.FeatureHasher(n_features=10, random_state=0)
        >>> X = [{'dog': 1, 'cat':2, 'elephant':4},{'dog': 2, 'run': 5}]
        >>> for x in X:
        ...     print(hasher.transform_one(x))
        defaultdict(<class 'int'>, {2: -1, 1: 2, 8: 4})
        defaultdict(<class 'int'>, {2: -2, 4: 5})

    """

    def __init__(self, n_features=1048576, alternate_sign=True, encoding='utf-8',
                 random_state=None):
        self.n_features = n_features
        self.alternate_sign = alternate_sign
        self.encoding = encoding
        self.rng = utils.check_random_state(random_state)
        self.seed = self.rng.randint(0, 2 ** 32 - 1)

    def fit_one(self, x, y=None):
        return self.transform_one(x)

    def hash_str(self, s):
        return utils.murmurhash3_32(bytes(s, self.encoding), self.seed)

    def transform_one(self, x):

        x_hashed = collections.defaultdict(int)

        for feature, value in x.items():

            if isinstance(value, str):
                feature = f'{feature}={value}'
                value = 1

            if value == 0:
                continue

            h = self.hash_str(feature)
            i = h % self.n_features

            if self.alternate_sign:
                value *= (h >= 0) * 2 - 1

            x_hashed[i] += value

        return x_hashed
