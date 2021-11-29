import collections

from river import base
from typing import Dict, Union

__all__ = ["OneHotEncoder", "OneHotListEncoder"]


class OneHotEncoder(base.Transformer):
    """One-hot encoding.

    This transformer will encode every feature it is provided it with. You can apply it to a
    subset of features by composing it with `compose.Select` or `compose.SelectType`.

    Parameters
    ----------
    sparse
        Whether or not 0s should be made explicit or not.

    Examples
    --------

    Let us first create an example dataset.

    >>> from pprint import pprint
    >>> import random
    >>> import string

    >>> random.seed(42)
    >>> alphabet = list(string.ascii_lowercase)
    >>> X = [
    ...     {
    ...         'c1': random.choice(alphabet),
    ...         'c2': random.choice(alphabet),
    ...     }
    ...     for _ in range(4)
    ... ]
    >>> pprint(X)
    [{'c1': 'u', 'c2': 'd'},
        {'c1': 'a', 'c2': 'x'},
        {'c1': 'i', 'c2': 'h'},
        {'c1': 'h', 'c2': 'e'}]

    We can now apply one-hot encoding. All the provided are one-hot encoded, there is therefore
    no need to specify which features to encode.

    >>> import river.preprocessing

    >>> oh = river.preprocessing.OneHotEncoder(sparse=True)
    >>> for x in X:
    ...     oh = oh.learn_one(x)
    ...     pprint(oh.transform_one(x))
    {'c1_u': 1, 'c2_d': 1}
    {'c1_a': 1, 'c2_x': 1}
    {'c1_i': 1, 'c2_h': 1}
    {'c1_h': 1, 'c2_e': 1}

    The `sparse` parameter can be set to `False` in order to include the values that are not
    present in the output.

    >>> oh = river.preprocessing.OneHotEncoder(sparse=False)
    >>> for x in X[:2]:
    ...     oh = oh.learn_one(x)
    ...     pprint(oh.transform_one(x))
    {'c1_u': 1, 'c2_d': 1}
    {'c1_a': 1, 'c1_u': 0, 'c2_d': 0, 'c2_x': 1}

    A subset of the features can be one-hot encoded by using an instance of `compose.Select`.

    >>> from river import compose

    >>> pp = compose.Select('c1') | river.preprocessing.OneHotEncoder()

    >>> for x in X:
    ...     pp = pp.learn_one(x)
    ...     pprint(pp.transform_one(x))
    {'c1_u': 1}
    {'c1_a': 1, 'c1_u': 0}
    {'c1_a': 0, 'c1_i': 1, 'c1_u': 0}
    {'c1_a': 0, 'c1_h': 1, 'c1_i': 0, 'c1_u': 0}

    You can preserve the `c2` feature by using a union:

    >>> pp = compose.Select('c1') | river.preprocessing.OneHotEncoder()
    >>> pp += compose.Select('c2')

    >>> for x in X:
    ...     pp = pp.learn_one(x)
    ...     pprint(pp.transform_one(x))
    {'c1_u': 1, 'c2': 'd'}
    {'c1_a': 1, 'c1_u': 0, 'c2': 'x'}
    {'c1_a': 0, 'c1_i': 1, 'c1_u': 0, 'c2': 'h'}
    {'c1_a': 0, 'c1_h': 1, 'c1_i': 0, 'c1_u': 0, 'c2': 'e'}

    """

    def __init__(self, sparse=False):
        self.sparse = sparse
        self.values = collections.defaultdict(set)

    def learn_one(self, x):
        for i, xi in x.items():
            self.values[i].add(xi)
        return self

    def transform_one(self, x, y=None):
        oh = {}

        # Add 0s
        if not self.sparse:
            oh = {f"{i}_{v}": 0 for i, values in self.values.items() for v in values}

        # Add 1s
        for i, xi in x.items():
            oh[f"{i}_{xi}"] = 1

        return oh


class OneHotListEncoder(OneHotEncoder):
    """One-hot list encoding.

    This special case of a OneHotEncoder extends existing functionality, in that it will
    allow lists (or sets) to be passed to the OneHotEncoder.

    Parameters
    ----------
    sparse
        Whether or not 0s should be made explicit or not.

    Examples
    --------

    Similar to the example of the OneHotEncoder, but now the values
    contain lists
    >>> from pprint import pprint
    >>> X = [{'c1': ['u', 'a'], 'c2': ['d']},
    ...     {'c1': ['a', 'b'], 'c2': ['x']},
    ...     {'c1': ['i'], 'c2': ['h', 'z']},
    ...     {'c1': ['h', 'b'], 'c2': ['e']}]

    We can now apply one-hot encoding of the values in the list. All the provided features
     are one-hot encoded, there is therefore no need to specify which features to encode.

    >>> import river.preprocessing

    >>> oh = river.preprocessing.OneHotListEncoder(sparse=True)
    >>> for x in X:
    ...     oh = oh.learn_one(x)
    ...     pprint(oh.transform_one(x))
    {'c1_a': 1, 'c1_u': 1, 'c2_d': 1}
    {'c1_a': 1, 'c1_b': 1, 'c2_x': 1}
    {'c1_i': 1, 'c2_h': 1, 'c2_z': 1}
    {'c1_b': 1, 'c1_h': 1, 'c2_e': 1}

    The `sparse` parameter can be set to `False` in order to include the values that are not
    present in the output.

    >>> oh = river.preprocessing.OneHotListEncoder(sparse=False)
    >>> for x in X[:2]:
    ...     oh = oh.learn_one(x)
    ...     pprint(oh.transform_one(x))
    {'c1_a': 1, 'c1_u': 1, 'c2_d': 1}
    {'c1_a': 1, 'c1_b': 1, 'c1_u': 0, 'c2_d': 0, 'c2_x': 1}

    You can preserve the `c2` feature by using a union:

    >>> from river import compose

    >>> pp = compose.Select('c1') | river.preprocessing.OneHotListEncoder()
    >>> pp += compose.Select('c2')

    >>> for x in X:
    ...     pp = pp.learn_one(x)
    ...     pprint(pp.transform_one(x))
    {'c1_a': 1, 'c1_u': 1, 'c2': ['d']}
    {'c1_a': 1, 'c1_b': 1, 'c1_u': 0, 'c2': ['x']}
    {'c1_a': 0, 'c1_b': 0, 'c1_i': 1, 'c1_u': 0, 'c2': ['h', 'z']}
    {'c1_a': 0, 'c1_b': 1, 'c1_h': 1, 'c1_i': 0, 'c1_u': 0, 'c2': ['e']}

    If you accidentally pass non list/set values to the encoder, it will not one-hot encode
     the values In the example below, see the 'c2' feature and the one-hot encoded
     outputs without this feature included.

    >>> X = [{'c1': ['u', 'a'], 'c2': 'd'},
    ...     {'c1': ['a', 'b'], 'c2': 'x'},
    ...     {'c1': ['i'], 'c2': 'zh'},
    ...     {'c1': ['h', 'b'], 'c2': 'e'}]
    >>> oh = river.preprocessing.OneHotListEncoder(sparse=True)
    >>> for x in X:
    ...     oh = oh.learn_one(x)
    ...     pprint(oh.transform_one(x))
    {'c1_a': 1, 'c1_u': 1}
    {'c1_a': 1, 'c1_b': 1}
    {'c1_i': 1}
    {'c1_b': 1, 'c1_h': 1}
    """
    def __init__(self, sparse=False):
        super().__init__(sparse=sparse)

    def learn_one(self, x: Dict[str, Union[set, list]]):
        for i, xi in x.items():

            # Similar to the OneHotEncoder class, but now
            # xi is a list or a set
            # If this is not the case, ignore
            if type(xi) not in [list, set]:
                continue

            # Here, it's at least a list or a set.
            # If it's empty, we don't have any elements and will
            # effectively skip it. Otherwise, add all the elements of the
            # list to the defaultdict
            for xj in xi:
                self.values[i].add(xj)

        return self

    def transform_one(self, x: Dict[str, Union[set, list]], y=None):
        oh = {}

        # Add 0s
        if not self.sparse:
            oh = {f"{i}_{v}": 0 for i, values in self.values.items() for v in values}

        # Add 1s
        for i, xi in x.items():

            if type(xi) not in [list, set]:
                continue

            for xj in xi:
                # i is the name of the variable
                # xi is the name of the value
                oh[f"{i}_{xj}"] = 1

        return oh
