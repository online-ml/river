import abc
import inspect
import types
import typing

from . import base


class Estimator(base.Base, abc.ABC):
    """An estimator."""

    @property
    def _supervised(self):
        """Indicates whether or not the estimator is supervised or not.

        This is useful internally for determining if an estimator expects to be provided with a `y`
        value in it's `learn_one` method. For instance we use this in a pipeline to know whether or
        not we should pass `y` to an estimator or not.

        """
        return True

    def __or__(self, other):
        """Merge with another Transformer into a Pipeline."""
        from .. import compose
        if isinstance(other, compose.Pipeline):
            return other.__ror__(self)
        return compose.Pipeline(self, other)

    def __ror__(self, other):
        """Merge with another Transformer into a Pipeline."""
        from .. import compose
        if isinstance(other, compose.Pipeline):
            return other.__or__(self)
        return compose.Pipeline(other, self)

    @property
    def _tags(self) -> typing.Dict[str, bool]:
        """Return the estimator's tags.

        Tags can be used to specify what kind of inputs an estimator is able to process. For
        instance, some estimators can handle text, whilst others don't. Inheriting from
        `base.Estimator` will imply a set of default tags which can be overridden by implementing
        the `_more_tags` property.

        TODO: this could be a cachedproperty.

        """

        try:
            tags = self._more_tags()
        except AttributeError:
            tags = set()

        for parent in self.__class__.__mro__:
            try:
                tags |= parent._more_tags(self)
            except AttributeError:
                pass

        return tags
