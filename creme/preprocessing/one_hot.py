from .. import base


__all__ = ['OneHotEncoder']


class OneHotEncoder(base.Transformer):
    """One-hot encoding.

    Attributes:
        on (str): Attribute to one-hot encode.
        sparse (bool): Whether or not 0s should be made explicit or not.

    Example:

        ::

            >>> import pprint
            >>> import string
            >>> import creme.preprocessing
            >>> import numpy as np

            >>> rng = np.random.RandomState(42)
            >>> alphabet = list(string.ascii_lowercase)
            >>> X = [{'letter': letter} for letter in rng.choice(alphabet, size=10)]

            >>> one_hot = creme.preprocessing.OneHotEncoder('letter', sparse=True)
            >>> for x in X:
            ...     print(one_hot.fit_one(x).transform_one(x))
            {'letter_g': 1}
            {'letter_t': 1}
            {'letter_o': 1}
            {'letter_k': 1}
            {'letter_h': 1}
            {'letter_u': 1}
            {'letter_g': 1}
            {'letter_z': 1}
            {'letter_s': 1}
            {'letter_w': 1}

            >>> one_hot = creme.preprocessing.OneHotEncoder('letter')
            >>> for letter in ['a', 'b', 'c']:
            ...     x = {'letter': letter}
            ...     pprint.pprint(one_hot.fit_one(x).transform_one(x))
            {'letter_a': 1}
            {'letter_a': 0, 'letter_b': 1}
            {'letter_a': 0, 'letter_b': 0, 'letter_c': 1}

    """

    def __init__(self, on, sparse=False):
        self.on = on
        self.sparse = sparse
        self.values = set()

    def fit_one(self, x, y=None):
        self.values.add(x[self.on])
        return self

    def transform_one(self, x, y=None):
        oh = {}
        if not self.sparse:
            oh = {f'{self.on}_{i}': 0 for i in self.values}
        oh[f'{self.on}_{x[self.on]}'] = 1
        return oh

    def __str__(self):
        return f'OneHotEncoder({self.on})'
