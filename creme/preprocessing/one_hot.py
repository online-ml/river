import collections
import typing

from creme import base


__all__ = ['OneHotEncoder']


class OneHotEncoder(base.Transformer):
    """One-hot encoding.

    This transformer will encode every feature it is provided it with. You can apply it to a
    subset of features by composing it with `compose.Select` or `compose.SelectType`.

    Parameters:
        sparse: Whether or not 0s should be made explicit or not.

    Example:

        Let us first create an example dataset.

        >>> from pprint import pprint
        >>> import random
        >>> import string

        >>> random.seed(42)
        >>> alphabet = list(string.ascii_lowercase)
        >>> X = [
        ...     {
        ...         'letter_1': random.choice(alphabet),
        ...         'letter_2': random.choice(alphabet),
        ...     }
        ...     for _ in range(4)
        ... ]
        >>> pprint(X)
        [{'letter_1': 'u', 'letter_2': 'd'},
         {'letter_1': 'a', 'letter_2': 'x'},
         {'letter_1': 'i', 'letter_2': 'h'},
         {'letter_1': 'h', 'letter_2': 'e'}]

        We can now apply one-hot encoding. All the provided are one-hot encoded, there is therefore
        no need to specify which features to encode.

        >>> import creme.preprocessing

        >>> oh = creme.preprocessing.OneHotEncoder(sparse=True)
        >>> for x in X:
        ...     oh = oh.fit_one(x)
        ...     pprint(oh.transform_one(x))
        {'letter_1_u': 1, 'letter_2_d': 1}
        {'letter_1_a': 1, 'letter_2_x': 1}
        {'letter_1_i': 1, 'letter_2_h': 1}
        {'letter_1_h': 1, 'letter_2_e': 1}

        The `sparse` parameter can be set to `False` in order to include the values that are not
        present in the output.

        >>> oh = creme.preprocessing.OneHotEncoder(sparse=False)
        >>> for x in X[:2]:
        ...     oh = oh.fit_one(x)
        ...     pprint(oh.transform_one(x))
        {'letter_1_u': 1, 'letter_2_d': 1}
        {'letter_1_a': 1, 'letter_1_u': 0, 'letter_2_d': 0, 'letter_2_x': 1}

        A subset of the features can be one-hot encoded by using an instance of `compose.Select`.

        >>> from creme import compose

        >>> pp = compose.Select('letter_1') | creme.preprocessing.OneHotEncoder()

        >>> for x in X:
        ...     pp = pp.fit_one(x)
        ...     pprint(pp.transform_one(x))
        {'letter_1_u': 1}
        {'letter_1_a': 1, 'letter_1_u': 0}
        {'letter_1_a': 0, 'letter_1_i': 1, 'letter_1_u': 0}
        {'letter_1_a': 0, 'letter_1_h': 1, 'letter_1_i': 0, 'letter_1_u': 0}

    """

    def __init__(self, sparse=False):
        self.sparse = sparse
        self.values = collections.defaultdict(set)

    def fit_one(self, x, y=None):
        for i, xi in x.items():
            self.values[i].add(xi)
        return self

    def transform_one(self, x, y=None):
        oh = {}

        # Add 0s
        if not self.sparse:
            oh = {f'{i}_{v}': 0 for i, values in self.values.items() for v in values}

        # Add 1s
        for i, xi in x.items():
            oh[f'{i}_{xi}'] = 1

        return oh
