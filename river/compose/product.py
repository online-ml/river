from __future__ import annotations

import functools
import itertools

import numpy as np
import pandas as pd

from river import utils

from . import union

__all__ = ["TransformerProduct"]


class TransformerProduct(union.TransformerUnion):
    """Computes interactions between the outputs of a set transformers.

    This is for when you want to add interaction terms between groups of features. It may also
    be used an alternative to `feature_extraction.PolynomialExtender` when the latter is overkill.

    Parameters
    ----------
    transformers
        Ideally, a list of (name, estimator) tuples. A name is automatically inferred if none is
        provided.

    Examples
    --------

    Let's say we have a certain set of features with two groups. In practice these may be different
    namespaces, such one for items and the other for users.

    >>> x = dict(
    ...     a=0, b=1,  # group 1
    ...     x=2, y=3   # group 2
    ... )

    We might want to add interaction terms between groups `('a', 'b')` and `('x', 'y')`, as so:

    >>> from pprint import pprint
    >>> from river.compose import Select, TransformerProduct

    >>> product = TransformerProduct(
    ...     Select('a', 'b'),
    ...     Select('x', 'y')
    ... )
    >>> pprint(product.transform_one(x))
    {'a*x': 0, 'a*y': 0, 'b*x': 2, 'b*y': 3}

    This can also be done with the following shorthand:

    >>> product = Select('a', 'b') * Select('x', 'y')
    >>> pprint(product.transform_one(x))
    {'a*x': 0, 'a*y': 0, 'b*x': 2, 'b*y': 3}

    If you want to include the original terms, you can do something like this:

    >>> group_1 = Select('a', 'b')
    >>> group_2 = Select('x', 'y')
    >>> product = group_1 + group_2 + group_1 * group_2
    >>> pprint(product.transform_one(x))
    {'a': 0, 'a*x': 0, 'a*y': 0, 'b': 1, 'b*x': 2, 'b*y': 3, 'x': 2, 'y': 3}

    """

    def __str__(self):
        return " * ".join(map(str, self.transformers.values()))

    def __repr__(self):
        return super().__repr__().replace("Union", "Product", 1)

    def __add__(self, other):
        from .union import TransformerUnion

        return TransformerUnion(self, other)

    def __mul__(self, other):
        return self._add_step(other)

    def transform_one(self, x):
        outputs = [t.transform_one(x) for t in self.transformers.values()]
        return {
            "*".join(combo): utils.math.prod(outputs[i][f] for i, f in enumerate(combo))
            for combo in itertools.product(*outputs)
        }

    def transform_many(self, X):
        outputs = [t.transform_many(X) for t in self.transformers.values()]

        def multiply(a, b):
            # Fast-track for sparse[uint8] * sparse[uint8]
            if a.dtype == pd.SparseDtype("uint8") and b.dtype == pd.SparseDtype("uint8"):
                return a & b
            # Fast-track for sparse[uint8] * numeric
            if a.dtype == pd.SparseDtype("uint8"):
                c = np.zeros_like(b)
                true_mask = a.eq(1)
                c[true_mask] = b[true_mask]
                return pd.Series(
                    c,
                    index=b.index,
                    dtype=pd.SparseDtype(b.dtype, fill_value=0),
                )
            # Fast-track for numeric * sparse[uint8]
            if b.dtype == pd.SparseDtype("uint8"):
                return multiply(b, a)
            # Fast-track for sparse * sparse
            if isinstance(a.dtype, pd.SparseDtype) and isinstance(b.dtype, pd.SparseDtype):
                return pd.Series(
                    a * b,
                    index=a.index,
                    dtype=pd.SparseDtype(
                        b.dtype, fill_value=a.sparse.fill_value * b.sparse.fill_value
                    ),
                )
            # Fast-track for sparse * numeric
            if isinstance(a.dtype, pd.SparseDtype):
                return pd.Series(
                    a * b, dtype=pd.SparseDtype(fill_value=a.sparse.fill_value, dtype=b.dtype)
                )
            # Fast-track for numeric * sparse
            if isinstance(b.dtype, pd.SparseDtype):
                return multiply(b, a)
            # Default
            return np.multiply(a, b)

        return pd.DataFrame(
            {
                "*".join(combo): functools.reduce(
                    multiply, (outputs[i][f] for i, f in enumerate(combo))
                )
                for combo in itertools.product(*outputs)
            },
            index=X.index,
        )
