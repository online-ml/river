import abc

from creme import base


class Transformer(base.Estimator):
    """A transformer."""

    def fit_one(self, x: dict) -> 'Transformer':
        """Update with a set of features `x`.

        A lot of transformers don't actually have to do anything during the `fit_one` step
        because they are stateless. For this reason the default behavior of this function is to do
        nothing. Transformers that however do something during the `fit_one` can override this
        method.

        Parameters:
            x: A dictionary of features.

        Returns:
            self

        """
        return self

    @abc.abstractmethod
    def transform_one(self, x: dict) -> dict:
        """Transform a set of features `x`.

        Parameters:
            x: A dictionary of features.

        Returns:
            dict

        """

    def __or__(self, other):
        """Merges with another Transformer into a Pipeline."""
        from .. import compose
        if isinstance(other, compose.Pipeline):
            return other.__ror__(self)
        return compose.Pipeline(self, other)

    def __ror__(self, other):
        """Merges with another Transformer into a Pipeline."""
        from .. import compose
        if isinstance(other, compose.Pipeline):
            return other.__or__(self)
        return compose.Pipeline(other, self)

    def __add__(self, other):
        """Merges with another Transformer into a TransformerUnion."""
        from .. import compose
        if isinstance(other, compose.TransformerUnion):
            return other.__add__(self)
        return compose.TransformerUnion(self, other)

    def __radd__(self, other):
        """Merges with another Transformer into a TransformerUnion."""
        from .. import compose
        if isinstance(other, compose.TransformerUnion):
            return other.__add__(self)
        return compose.TransformerUnion(other, self)


class SupervisedTransformer(Transformer):

    def fit_one(self, x: dict, y: base.typing.Target) -> 'SupervisedTransformer':
        """Update with a set of features `x` and a target `y`.

        Parameters:
            x: A dictionary of features.

        Returns:
            self

        """
        return self
