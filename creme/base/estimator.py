import abc
import copy
import inspect
import sys
import types
import typing

from .. import utils


DEFAULT_TAGS = {
    'handles_text': False,
    'requires_positive_data': False,
    'handles_categorical_features': False
}


class Estimator(abc.ABC):
    """An estimator."""

    def _is_supervised(self):
        """Indicates whether or not the estimator is supervised or not.

        This is useful internally for determining if an estimator expects to be provided with a `y`
        value in it's `fit_one` method. For instance we use this in a pipeline to know whether or
        not we should pass `y` to an estimator or not.

        """
        return True

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return _repr_obj(obj=self, params=self._get_params())

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

    def _get_params(self) -> typing.Dict[str, typing.Any]:
        return {
            name: getattr(self, name)
            for name, param in inspect.signature(self.__init__).parameters.items()  # type: ignore
            if param.kind != param.VAR_KEYWORD
        }

    def _set_params(self, new_params: typing.Optional[typing.Dict[str, typing.Any]] = None) -> 'Estimator':
        """Returns a new instance with the current parameters as well as new ones.

        The algorithm will be recursively called down `Pipeline`s and `TransformerUnion`s.

        Example:

            >>> from creme import linear_model
            >>> from creme import optim
            >>> from creme import preprocessing

            >>> model = (
            ...     preprocessing.StandardScaler() |
            ...     linear_model.LinearRegression(
            ...         optimizer=optim.SGD(lr=0.042),
            ...     )
            ... )

            >>> new_params = {
            ...     'LinearRegression': {
            ...         'l2': .001
            ...     }
            ... }

            >>> model._set_params(new_params)
            Pipeline (
              StandardScaler (),
              LinearRegression (
                optimizer=SGD (
                  lr=Constant (
                    learning_rate=0.042
                  )
                )
                loss=Squared ()
                l2=0.001
                intercept=0.
                intercept_lr=Constant (
                  learning_rate=0.01
                )
                clip_gradient=1e+12
                initializer=Zeros ()
              )
            )

        """

        if new_params is None:
            new_params = {}

        params = {
            **self._get_params(),
            **new_params
        }

        return self.__class__(**copy.deepcopy(params))  # type: ignore

    @property
    def _tags(self) -> typing.Dict[str, bool]:
        """Returns the estimator's tags.

        Tags can be used to specify what kind of inputs an estimator is able to process. For
        instance, some estimators can handle text, whilst others require positive numeric data.
        Inheriting from `base.Estimator` will imply a set of default tags which can be overriden
        by implementing the `base.Estimator._more_tags` method.

        """

        tags: typing.Dict[str, bool] = {}

        for base_class in inspect.getmro(self.__class__)[1:]:
            if isinstance(base_class, Estimator):
                _update_if_consistent(tags, base_class._more_tags)

        _update_if_consistent(tags, self._more_tags)

        return {**DEFAULT_TAGS, **tags}

    @property
    def _more_tags(self) -> typing.Dict[str, bool]:
        """Specific tags for this estimator."""
        return {}

    @property
    def _memory_usage(self) -> str:
        """Returns the memory usage in a human readable format."""

        def get_size(obj, seen=None):
            """Recursively finds size of objects"""
            size = sys.getsizeof(obj)
            if seen is None:
                seen = set()
            obj_id = id(obj)
            if obj_id in seen:
                return 0
            # Important mark as seen *before* entering recursion to gracefully handle
            # self-referential objects
            seen.add(obj_id)
            if isinstance(obj, dict):
                size += sum([get_size(v, seen) for v in obj.values()])
                size += sum([get_size(k, seen) for k in obj.keys()])
            elif hasattr(obj, '__dict__'):
                size += get_size(vars(obj), seen)
            elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
                size += sum([get_size(i, seen) for i in obj])
            return size

        mem_usage = get_size(self)
        return utils.pretty.humanize_bytes(mem_usage)


def _update_if_consistent(dict1: dict, dict2: dict):
    common_keys = set(dict1.keys()).intersection(dict2.keys())
    for key in common_keys:
        if dict1[key] != dict2[key]:
            raise TypeError(
                f'Inconsistent values for tag {key}: {dict1[key]} != {dict2[key]}')
    dict1.update(dict2)


def _repr_obj(obj, params=None, show_modules: bool = False, depth: int = 0) -> str:
    """Returns a pretty representation of an object."""

    rep = f'{obj.__class__.__name__} ('
    if show_modules:
        rep = f'{obj.__class__.__module__}.{rep}'
    tab = '\t'

    if params is None:
        params = {
            name: getattr(obj, name)
            for name, param in inspect.signature(obj.__init__).parameters.items()  # type: ignore
            if not (
                param.name == 'args' and param.kind == param.VAR_POSITIONAL or
                param.name == 'kwargs' and param.kind == param.VAR_KEYWORD
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
                f'{val:.0e}'
                if (val > 1e5 or (val < 1e-4 and val > 0)) else
                f'{val:.6f}'.rstrip('0')
            )
        elif isinstance(val, set):
            val = sorted(val)
        elif hasattr(val, '__class__') and 'creme.' in str(type(val)):
            val = _repr_obj(obj=val, show_modules=show_modules, depth=depth + 1)

        rep += f'\n{tab * (depth + 1)}{name}={val}'

    if n_params:
        rep += f'\n{tab * depth}'
    rep += ')'

    return rep.expandtabs(2)
