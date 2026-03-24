from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from river import base

from .estimator import Estimator  # Prevent a circular import of module base

T = TypeVar("T", bound=Estimator)


class Wrapper(ABC, Generic[T]):
    """A wrapper model."""

    @property
    @abstractmethod
    def _wrapped_model(self) -> T:
        """Provides access to the wrapped model."""

    @property
    def _labelloc(self) -> str:
        """Indicates location of the wrapper name when drawing pipelines."""
        return "t"  # for top

    def __str__(self) -> str:
        return f"{type(self).__name__}({self._wrapped_model})"

    def _more_tags(self) -> set[str]:
        return self._wrapped_model._tags

    @property
    def _supervised(self) -> bool:
        return self._wrapped_model._supervised

    @property
    def _multiclass(self) -> bool:
        return isinstance(self._wrapped_model, base.Classifier) and self._wrapped_model._multiclass
