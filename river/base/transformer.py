from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any

from river import base

if TYPE_CHECKING:
    from river import compose


class Transformer(base.Estimator, abc.ABC):
    def __add__(self, other: Transformer) -> compose.TransformerUnion:
        """Fuses with another Transformer into a TransformerUnion."""
        from river import compose

        return compose.TransformerUnion(self, other)

    def __radd__(self, other: Transformer) -> compose.TransformerUnion:
        """Fuses with another Transformer into a TransformerUnion."""
        from river import compose

        return compose.TransformerUnion(other, self)

    def __mul__(
        self,
        other: Transformer
        | compose.Pipeline
        | base.typing.FeatureName
        | list[base.typing.FeatureName],
    ) -> compose.Grouper | compose.TransformerProduct:
        from river import compose

        if isinstance(other, Transformer) or isinstance(other, compose.Pipeline):
            return compose.TransformerProduct(self, other)

        return compose.Grouper(transformer=self, by=other)

    def __rmul__(
        self,
        other: Transformer
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

    @property
    def _supervised(self) -> bool:
        return False
