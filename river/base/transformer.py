import abc

from river import base


class Transformer(base.Estimator):
    """A transformer."""

    @property
    def _supervised(self):
        return False

    def learn_one(self, x: dict, **kwargs) -> "Transformer":
        """Update with a set of features `x`.

        A lot of transformers don't actually have to do anything during the `learn_one` step
        because they are stateless. For this reason the default behavior of this function is to do
        nothing. Transformers that however do something during the `learn_one` can override this
        method.

        Parameters
        ----------
        x
            A dictionary of features.
        kwargs
            Some models might allow/require providing extra parameters, such as sample weights.

        Returns
        -------
        self

        """
        return self

    @abc.abstractmethod
    def transform_one(self, x: dict) -> dict:
        """Transform a set of features `x`.

        Parameters
        ----------
        x
            A dictionary of features.

        Returns
        -------
        The transformed values.

        """

    def __add__(self, other):
        """Fuses with another Transformer into a TransformerUnion."""
        from .. import compose

        if isinstance(other, compose.TransformerUnion):
            return other.__add__(self)
        return compose.TransformerUnion(self, other)

    def __radd__(self, other):
        """Fuses with another Transformer into a TransformerUnion."""
        from .. import compose

        if isinstance(other, compose.TransformerUnion):
            return other.__add__(self)
        return compose.TransformerUnion(other, self)

    def __mul__(self, feature):
        """Creates a Grouper."""
        from .. import compose

        return compose.Grouper(transformer=self, by=feature)

    def __rmul__(self, feature):
        """Creates a Grouper."""
        from .. import compose

        return compose.Grouper(transformer=self, by=feature)


class SupervisedTransformer(Transformer):
    """A supervised transformer."""

    @property
    def _supervised(self):
        return True

    def learn_one(self, x: dict, y: base.typing.Target, **kwargs) -> "SupervisedTransformer":
        """Update with a set of features `x` and a target `y`.

        Parameters
        ----------
        x
            A dictionary of features.
        kwargs
            Some models might allow/require providing extra parameters, such as sample weights.

        Returns
        -------
        self

        """
        return self
