from __future__ import annotations

import abc
import typing
from typing import Any

from river import base

if typing.TYPE_CHECKING:
    import pandas as pd

    from river import compose


class BaseTransformer:
    def __add__(self, other: BaseTransformer) -> compose.TransformerUnion:
        """Fuses with another Transformer into a TransformerUnion."""
        from river import compose

        return compose.TransformerUnion(self, other)

    def __radd__(self, other: BaseTransformer) -> compose.TransformerUnion:
        """Fuses with another Transformer into a TransformerUnion."""
        from river import compose

        return compose.TransformerUnion(other, self)

    def __mul__(
        self,
        other: BaseTransformer
        | compose.Pipeline
        | base.typing.FeatureName
        | list[base.typing.FeatureName],
    ) -> compose.Grouper | compose.TransformerProduct:
        from river import compose

        if isinstance(other, BaseTransformer) or isinstance(other, compose.Pipeline):
            return compose.TransformerProduct(self, other)

        return compose.Grouper(transformer=self, by=other)

    def __rmul__(
        self,
        other: BaseTransformer
        | compose.Pipeline
        | base.typing.FeatureName
        | list[base.typing.FeatureName],
    ) -> compose.Grouper | compose.TransformerProduct:
        """Creates a Grouper."""
        return self * other

    @abc.abstractmethod
    def transform_one(
        self, x: dict[base.typing.FeatureName, Any]
    ) -> dict[base.typing.FeatureName, Any]:
        """Transform a set of features `x`.

        Parameters
        ----------
        x
            A dictionary of features.

        Returns
        -------
        The transformed values.

        """


class Transformer(base.Estimator, BaseTransformer):
    """A transformer."""

    @property
    def _supervised(self) -> bool:
        return False

    def learn_one(self, x: dict[base.typing.FeatureName, Any]) -> None:
        """Update with a set of features `x`.

        A lot of transformers don't actually have to do anything during the `learn_one` step
        because they are stateless. For this reason the default behavior of this function is to do
        nothing. Transformers that however do something during the `learn_one` can override this
        method.

        Parameters
        ----------
        x
            A dictionary of features.

        """
        return


class SupervisedTransformer(base.Estimator, BaseTransformer):
    """A supervised transformer."""

    @property
    def _supervised(self) -> bool:
        return True

    def learn_one(self, x: dict[base.typing.FeatureName, Any], y: base.typing.Target) -> None:
        """Update with a set of features `x` and a target `y`.

        Parameters
        ----------
        x
            A dictionary of features.
        y
            A target.

        """
        return


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

    def learn_many(self, X: pd.DataFrame) -> None:
        """Update with a mini-batch of features.

        A lot of transformers don't actually have to do anything during the `learn_many` step
        because they are stateless. For this reason the default behavior of this function is to do
        nothing. Transformers that however do something during the `learn_many` can override this
        method.

        Parameters
        ----------
        X
            A DataFrame of features.

        """
        return


class MiniBatchSupervisedTransformer(Transformer):
    """A supervised transformer that can operate on mini-batches."""

    @property
    def _supervised(self) -> bool:
        return True

    @abc.abstractmethod
    def learn_many(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Update the model with a mini-batch of features `X` and targets `y`.

        Parameters
        ----------
        X
            A dataframe of features.
        y
            A series of boolean target values.

        """
        return

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
