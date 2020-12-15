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


def _repr_obj(obj, params=None, show_modules: bool = False, depth: int = 0) -> str:
    """Return a pretty representation of an object."""

    rep = f"{obj.__class__.__name__} ("
    if show_modules:
        rep = f"{obj.__class__.__module__}.{rep}"
    tab = "\t"

    if params is None:
        params = {
            name: getattr(obj, name)
            for name, param in inspect.signature(obj.__init__).parameters.items()  # type: ignore
            if not (
                param.name == "args"
                and param.kind == param.VAR_POSITIONAL
                or param.name == "kwargs"
                and param.kind == param.VAR_KEYWORD
            )
        }

    n_params = 0

    for name, val in params.items():

        n_params += 1

        # Prettify the attribute when applicable
        if isinstance(val, types.FunctionType):
            val = val.__name__
        if isinstance(val, str):
            val = f'"{val}"'
        elif isinstance(val, float):
            val = (
                f"{val:.0e}"
                if (val > 1e5 or (val < 1e-4 and val > 0))
                else f"{val:.6f}".rstrip("0")
            )
        elif isinstance(val, set):
            val = sorted(val)
        elif hasattr(val, "__class__") and "river." in str(type(val)):
            val = _repr_obj(obj=val, show_modules=show_modules, depth=depth + 1)

        rep += f"\n{tab * (depth + 1)}{name}={val}"

    if n_params:
        rep += f"\n{tab * depth}"
    rep += ")"

    return rep.expandtabs(2)
