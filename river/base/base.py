import copy
import inspect
import sys
import types
import typing


class Base:
    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return _repr_obj(obj=self, params=self._get_params())

    @classmethod
    def _unit_test_params(cls):
        """Instantiates an object with default arguments.

        Most parameters of each object have a default value. However, this isn't always the case,
        in particular for meta-models where the wrapped model is typically not given a default
        value. It's useful to have a default value set for testing reasons, which is the purpose of
        this method. By default it simply calls the __init__ function. It may be overridden on an
        individual as needed.

        """
        return {}

    def _get_params(self) -> typing.Dict[str, typing.Any]:
        """Return the parameters that were used during initialization."""

        params = {}

        for name, param in inspect.signature(self.__init__).parameters.items():
            if param.kind == param.VAR_KEYWORD:  # handles **kwargs
                params.update(getattr(self, name))
            else:
                params[name] = getattr(self, name)

        return params

    def _set_params(
        self, new_params: typing.Optional[typing.Dict[str, typing.Any]] = None
    ) -> "Base":
        """Return a new instance with the current parameters as well as new ones.

        Calling this without any parameters will essentially clone the estimator.

        The algorithm will be recursively called down `Pipeline`s and `TransformerUnion`s.

        Examples
        --------

        >>> from river import linear_model
        >>> from river import optim
        >>> from river import preprocessing

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
            intercept_init=0.
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

        params = {**self._get_params(), **new_params}

        for name, param in params.items():
            if isinstance(param, Base):
                params[name] = param.clone()
            else:
                params[name] = copy.deepcopy(param)

        return self.__class__(**params)  # type: ignore

    def clone(self):
        """Return a fresh estimator with the same parameters.

        The clone has the same parameters but has not been updated with any data.

        This works by looking at the parameters from the class signature. Each parameter is either

        - recursively cloned if it's a River classes.
        - deep-copied via `copy.deepcopy` if not.

        If the calling object is stochastic (i.e. it accepts a seed parameter) and has not been
        seeded, then the clone will not be idempotent. Indeed, this method's purpose if simply to
        return a new instance with the same input parameters.

        """
        return self._set_params()

    @property
    def _memory_usage_raw(self) -> int:
        """Return the memory usage in bytes."""

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
            elif hasattr(obj, "__dict__"):
                size += get_size(vars(obj), seen)
            elif hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes, bytearray)):
                size += sum([get_size(i, seen) for i in obj])
            return size

        return get_size(self)

    @property
    def _memory_usage(self) -> str:
        """Return the memory usage in a human readable format."""
        from river import utils

        return utils.pretty.humanize_bytes(self._memory_usage_raw)


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
