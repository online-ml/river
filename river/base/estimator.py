import abc
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

    @classmethod
    def _unit_test_params(self):
        """Indicates which parameters to use during unit testing.

        Most estimators have a default value for each of their parameters. However, in some cases,
        no default value is specified. This class method allows to circumvent this issue when the
        model has to be instantiated during unit testing.

        This can also be used to override default parameters that are computationally expensive,
        such as the number of base models in an ensemble.

        """
        return {}

    def _unit_test_skips(self):
        """Indicates which checks to skip during unit testing.

        Most estimators pass the full test suite. However, in some cases, some estimators might not
        be able to pass certain checks.

        """
        return set()
