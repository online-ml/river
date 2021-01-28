import collections

from river import base

__all__ = ["OneHotEncoder"]


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
