import itertools

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

    def __repr__(self):
        return super().__repr__().replace("Union", "Product", 1)

    def __mul__(self, other):
        return self._add_step(other)

    def transform_one(self, x):
        """Passes the data through each transformer and packs the results together."""
        outputs = [t.transform_one(x) for t in self.transformers.values()]
        return {
            "*".join(combo): utils.math.prod(outputs[i][f] for i, f in enumerate(combo))
            for combo in itertools.product(*outputs)
        }
