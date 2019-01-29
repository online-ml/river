from .. import base


__all__ = ['OneHotEncoder']


class OneHotEncoder(base.Transformer):
    """One-hot encoding.

    Example:

        >>> import pprint as pp
        >>> import string
        >>> import creme.preprocessing
        >>> import numpy as np

        >>> rng = np.random.RandomState(42)
        >>> alphabet = list(string.ascii_lowercase)
        >>> X = [{'letter': letter} for letter in rng.choice(alphabet, size=10)]

        >>> one_hot = creme.preprocessing.OneHotEncoder()
        >>> pp.pprint([one_hot.fit_one(x) for x in X])
        [{'letter_g': True},
         {'letter_t': True},
         {'letter_o': True},
         {'letter_k': True},
         {'letter_h': True},
         {'letter_u': True},
         {'letter_g': True},
         {'letter_z': True},
         {'letter_s': True},
         {'letter_w': True}]

    """

    def _one_hot(self, x):
        for i, xi in x.items():
            if isinstance(xi, str):
                x[f'{i}_{x.pop(i)}'] = True
        return x

    def fit_one(self, x, y=None):
        return self._one_hot(x)

    def transform_one(self, x, y=None):
        return self._one_hot(x)
