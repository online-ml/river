import abc

import pandas as pd

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
        from river import compose

        if isinstance(other, compose.TransformerUnion):
            return other.__add__(self)
        return compose.TransformerUnion(self, other)

    def __radd__(self, other):
        """Fuses with another Transformer into a TransformerUnion."""
        from river import compose

        if isinstance(other, compose.TransformerUnion):
            return other.__add__(self)
        return compose.TransformerUnion(other, self)

    def __mul__(self, other):
        from river import compose

        if isinstance(other, (Transformer, compose.Pipeline)):
            return compose.TransformerProduct(self, other)

        return compose.Grouper(transformer=self, by=other)

    def __rmul__(self, other):
        """Creates a Grouper."""
        return self * other


class SupervisedTransformer(Transformer):
    """A supervised transformer."""

    @property
    def _supervised(self):
        return True

    def learn_one(
        self, x: dict, y: base.typing.Target, **kwargs
    ) -> "SupervisedTransformer":
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


class MiniBatchTransformer(Transformer):
    """A transform that can operate on mini-batches."""

    @abc.abstractmethod
    def transform_many(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform a mini-batch of features.

        Parameters
        ----------
        X
            A DataFrame of features.

        Returns
        -------
        A new DataFrame.

        """

    def learn_many(self, X: pd.DataFrame, **kwargs) -> "Transformer":
        """Update with a mini-batch of features.

        A lot of transformers don't actually have to do anything during the `learn_many` step
        because they are stateless. For this reason the default behavior of this function is to do
        nothing. Transformers that however do something during the `learn_many` can override this
        method.

        Parameters
        ----------
        X
            A DataFrame of features.
        kwargs
            Some models might allow/require providing extra parameters, such as sample weights.

        Returns
        -------
        self

        """
        return self
