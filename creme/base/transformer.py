import abc

from creme import base


class Transformer(base.Estimator):
    """A transformer."""

    def fit_one(self, x: dict) -> 'Transformer':
        """Fits to a set of features `x` and an optional target `y`.

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
        """Transforms a set of features `x`.

        Parameters:
            x: A dictionary of features.

        Returns:
            dict

        """

    @property
    def is_supervised(self) -> bool:
        """Indicates if the transformer uses the target `y` or not.

        Supervised transformers have to be handled differently from unsupervised transformers in an
        online setting. This is especially true for target encoding where leakage can easily occur.
        Most transformers are unsupervised and so this property returns by default `False`.
        Transformers that are supervised must override this property in their definition.

        """
        return False

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
    """A supervised transformer.

    Supervised transformers have to be handled differently from unsupervised transformers in an
    online setting. This is especially true for target encoding where leakage can easily occur.

    """

    def fit_one(self, x: dict, y: base.typing.Target) -> 'SupervisedTransformer':
        """Fits to a set of features `x` and an optional target `y`.

        Parameters:
            x: A dictionary of features.
            y: A target label.

        Returns:
            self

        """
        return self
