from __future__ import annotations

from abc import ABC, abstractmethod


class Wrapper(ABC):
    """A wrapper model."""

    @property
    @abstractmethod
    def _wrapped_model(self):
        """Provides access to the wrapped model."""

    @property
    def _labelloc(self):
        """Indicates location of the wrapper name when drawing pipelines."""
        return "t"  # for top

    def __str__(self):
        return f"{type(self).__name__}({self._wrapped_model})"

    def _more_tags(self):
        return self._wrapped_model._tags

    @property
    def _supervised(self):
        return self._wrapped_model._supervised

    @property
    def _multiclass(self):
        return self._wrapped_model._multiclass
