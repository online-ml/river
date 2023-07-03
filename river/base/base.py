from __future__ import annotations

import collections
import contextlib
import copy
import inspect
import itertools
import logging
import sys
import types
import typing


class Base:
    """Base class that is inherited by the majority of classes in River.

    This base class allows us to handle the following tasks in a uniform manner:

    - Getting and setting parameters
    - Displaying information
    - Mutating/cloning

    """

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return _repr_obj(obj=self)

    @classmethod
    def _unit_test_params(cls):
        """Instantiates an object with default arguments.

        Most parameters of each object have a default value. However, this isn't always the case,
        in particular for meta-models where the wrapped model is typically not given a default
        value. It's useful to have a default value set for testing reasons, which is the purpose of
        this method. By default it simply calls the __init__ function. It may be overridden on an
        individual as needed.

        """
        yield {}

    def _get_params(self) -> dict[str, typing.Any]:
        """Return the parameters that were used during initialization."""

        params = {}

        for name, param in inspect.signature(self.__init__).parameters.items():  # type: ignore
            # *args
            if param.kind == param.VAR_POSITIONAL:
                if positional_args := getattr(self, name, []):
                    params["_POSITIONAL_ARGS"] = positional_args
                continue

            # **kwargs
            if param.kind == param.VAR_KEYWORD:
                for k, v in getattr(self, name, {}).items():
                    if isinstance(v, Base):
                        params[k] = (v.__class__, v._get_params())
                    else:
                        params[k] = v
                continue

            # Keywords parameters
            attr = getattr(self, name)
            if isinstance(attr, Base):
                params[name] = (attr.__class__, attr._get_params())
            else:
                params[name] = attr

        return params

    def clone(self, new_params: dict | None = None, include_attributes=False):
        """Return a fresh estimator with the same parameters.

        The clone has the same parameters but has not been updated with any data.

        This works by looking at the parameters from the class signature. Each parameter is either

        - recursively cloned if its a class.
        - deep-copied via `copy.deepcopy` if not.

        If the calling object is stochastic (i.e. it accepts a seed parameter) and has not been
        seeded, then the clone will not be idempotent. Indeed, this method's purpose if simply to
        return a new instance with the same input parameters.

        Parameters
        ----------
        new_params
        include_attributes
            Whether attributes that are not present in the class' signature should also be cloned
            or not.

        Examples
        --------

        >>> from river import linear_model
        >>> from river import optim

        >>> model = linear_model.LinearRegression(
        ...     optimizer=optim.SGD(lr=0.042),
        ... )

        >>> new_params = {
        ...     'optimizer': optim.SGD(.001)
        ... }

        >>> model.clone(new_params)
        LinearRegression (
          optimizer=SGD (
            lr=Constant (
              learning_rate=0.001
            )
          )
          loss=Squared ()
          l2=0.
          l1=0.
          intercept_init=0.
          intercept_lr=Constant (
            learning_rate=0.01
          )
          clip_gradient=1e+12
          initializer=Zeros ()
        )

        The algorithm is recursively called down `Pipeline`s and `TransformerUnion`s.

        >>> from river import compose
        >>> from river import preprocessing

        >>> model = compose.Pipeline(
        ...     preprocessing.StandardScaler(),
        ...     linear_model.LinearRegression(
        ...         optimizer=optim.SGD(0.042),
        ...     )
        ... )

        >>> new_params = {
        ...     'LinearRegression': {
        ...         'optimizer': optim.SGD(0.03)
        ...     }
        ... }

        >>> model.clone(new_params)
        Pipeline (
          StandardScaler (
            with_std=True
          ),
          LinearRegression (
            optimizer=SGD (
              lr=Constant (
                learning_rate=0.03
              )
            )
            loss=Squared ()
            l2=0.
            l1=0.
            intercept_init=0.
            intercept_lr=Constant (
              learning_rate=0.01
            )
            clip_gradient=1e+12
            initializer=Zeros ()
          )
        )

        """

        def is_class_param(param):
            # See expand_param_grid to understand why this is necessary
            return (
                isinstance(param, tuple)
                and inspect.isclass(param[0])
                and isinstance(param[1], dict)
            )

        # Override the default parameters with the new ones
        params = self._get_params()
        params.update(new_params or {})

        # Clone by recursing
        clone = self.__class__(
            *(params.get("_POSITIONAL_ARGS", [])),
            **{
                name: (
                    getattr(self, name).clone(param[1])
                    if is_class_param(param)
                    else copy.deepcopy(param)
                )
                for name, param in params.items()
                if name != "_POSITIONAL_ARGS"
            },
        )

        if not include_attributes:
            return clone

        for attr, value in self.__dict__.items():
            if attr not in params:
                setattr(clone, attr, copy.deepcopy(value))
        return clone

    @property
    def _mutable_attributes(self) -> set:
        return set()

    def mutate(self, new_attrs: dict):
        """Modify attributes.

        This changes parameters inplace. Although you can change attributes yourself, this is the
        recommended way to proceed. By default, all attributes are immutable, meaning they
        shouldn't be mutated. Calling `mutate` on an immutable attribute raises a `ValueError`.
        Mutable attributes are specified via the `_mutable_attributes` property, and are thus
        specified on a per-estimator basis.

        Parameters
        ----------
        new_attrs

        Examples
        --------

        >>> from river import linear_model
        >>> from river import optim

        >>> model = linear_model.LinearRegression(
        ...     optimizer=optim.SGD(0.042),
        ... )

        >>> new_params = {
        ...     'optimizer': {'lr': optim.schedulers.Constant(0.001)}
        ... }

        >>> model.mutate(new_params)
        >>> model
        LinearRegression (
          optimizer=SGD (
            lr=Constant (
              learning_rate=0.001
            )
          )
          loss=Squared ()
          l2=0.
          l1=0.
          intercept_init=0.
          intercept_lr=Constant (
            learning_rate=0.01
          )
          clip_gradient=1e+12
          initializer=Zeros ()
        )

        The algorithm is recursively called down `Pipeline`s and `TransformerUnion`s.

        >>> from river import compose
        >>> from river import preprocessing

        >>> model = compose.Pipeline(
        ...     preprocessing.StandardScaler(),
        ...     linear_model.LinearRegression(
        ...         optimizer=optim.SGD(lr=0.042),
        ...     )
        ... )

        >>> new_params = {
        ...     'LinearRegression': {
        ...         'l2': 5,
        ...         'optimizer': {'lr': optim.schedulers.Constant(0.03)}
        ...     }
        ... }

        >>> model.mutate(new_params)
        >>> model
        Pipeline (
          StandardScaler (
            with_std=True
          ),
          LinearRegression (
            optimizer=SGD (
              lr=Constant (
                learning_rate=0.03
              )
            )
            loss=Squared ()
            l2=5
            l1=0.
            intercept_init=0.
            intercept_lr=Constant (
              learning_rate=0.01
            )
            clip_gradient=1e+12
            initializer=Zeros ()
          )
        )

        """

        def _mutate(obj, new_attrs):
            def is_class_attr(name, attr):
                return hasattr(getattr(obj, name), "mutate") and isinstance(attr, dict)

            for name, attr in new_attrs.items():
                if not hasattr(obj, name):
                    raise ValueError(f"'{name}' is not an attribute of {obj.__class__.__name__}")

                # Check the attribute is mutable
                if name not in obj._mutable_attributes:
                    raise ValueError(
                        f"'{name}' is not a mutable attribute of {obj.__class__.__name__}"
                    )

                if is_class_attr(name, attr):
                    _mutate(obj=getattr(obj, name), new_attrs=attr)
                else:
                    setattr(obj, name, attr)

        _mutate(obj=self, new_attrs=new_attrs)

    @property
    def _is_stochastic(self):
        """Indicates if the model contains an unset seed parameter.

        The convention in River is to control randomness by exposing a seed parameter. This seed
        typically defaults to `None`. If the seed is set to `None`, then the model is expected to
        produce non-reproducible results. In other words it is not deterministic and is instead
        stochastic. This method checks if this is the case by looking for a None `seed` in the
        model's parameters.

        """

        def is_class_param(param):
            return (
                isinstance(param, tuple)
                and inspect.isclass(param[0])
                and isinstance(param[1], dict)
            )

        def find(params):
            if not isinstance(params, dict):
                return False
            for name, param in params.items():
                if name == "seed" and param is None:
                    return True
                if is_class_param(param) and find(param[1]):
                    return True
            return False

        return find(self._get_params())

    @property
    def _raw_memory_usage(self) -> int:
        """Return the memory usage in bytes."""

        import numpy as np

        buffer = collections.deque([self])
        seen = set()
        size = 0
        while len(buffer) > 0:
            obj = buffer.popleft()
            obj_id = id(obj)
            if obj_id in seen:
                continue
            size += sys.getsizeof(obj)
            # Important mark as seen to gracefully handle self-referential objects
            seen.add(obj_id)
            if isinstance(obj, dict):
                buffer.extend([k for k in obj.keys()])
                buffer.extend([v for v in obj.values()])
            elif hasattr(obj, "__dict__"):  # Save object contents
                contents: dict = vars(obj)
                size += sys.getsizeof(contents)
                buffer.extend([k for k in contents.keys()])
                buffer.extend([v for v in contents.values()])
            elif isinstance(obj, np.ndarray):
                size += obj.nbytes
            elif (
                isinstance(obj, itertools.count)
                or isinstance(obj, itertools.cycle)
                or isinstance(obj, itertools.repeat)
            ):
                ...
            elif hasattr(obj, "__iter__") and not (
                isinstance(obj, str) or isinstance(obj, bytes) or isinstance(obj, bytearray)
            ):
                buffer.extend([i for i in obj])  # type: ignore

        return size

    @property
    def _memory_usage(self) -> str:
        """Return the memory usage in a human readable format."""
        from river import utils

        return utils.pretty.humanize_bytes(self._raw_memory_usage)


def _log_method_calls(self, name, class_condition, method_condition):
    method = object.__getattribute__(self, name)
    if (
        not name.startswith("_")
        and inspect.ismethod(method)
        and class_condition(self)
        and method_condition(method)
    ):
        logging.debug(f"{self.__class__.__name__}.{name}")
    return method


@contextlib.contextmanager
def log_method_calls(
    class_condition: typing.Callable[[typing.Any], bool] | None = None,
    method_condition: typing.Callable[[typing.Any], bool] | None = None,
):
    """A context manager to log method calls.

    All method calls will be logged by default. This behavior can be overriden by passing filtering
    functions.

    Parameters
    ----------
    class_condition
        A function which determines if a class should be logged or not.
    method_condition
        A function which determines if a method should be logged or not.

    Examples
    --------
    >>> import io
    >>> import logging
    >>> from river import anomaly
    >>> from river import compose
    >>> from river import datasets
    >>> from river import preprocessing
    >>> from river import utils

    >>> model = compose.Pipeline(
    ...     preprocessing.MinMaxScaler(),
    ...     anomaly.HalfSpaceTrees(seed=42)
    ... )

    >>> class_condition = lambda x: x.__class__.__name__ in ('MinMaxScaler', 'HalfSpaceTrees')

    >>> logger = logging.getLogger()
    >>> logger.setLevel(logging.DEBUG)

    >>> logs = io.StringIO()
    >>> sh = logging.StreamHandler(logs)
    >>> sh.setLevel(logging.DEBUG)
    >>> logger.addHandler(sh)

    >>> with utils.log_method_calls(class_condition):
    ...     for x, y in datasets.CreditCard().take(1):
    ...         score = model.score_one(x)
    ...         model = model.learn_one(x)

    >>> print(logs.getvalue())
    MinMaxScaler.transform_one
    HalfSpaceTrees.score_one
    MinMaxScaler.learn_one
    MinMaxScaler.transform_one
    HalfSpaceTrees.learn_one

    >>> logs.close()

    """
    old = Base.__getattribute__
    class_condition = class_condition or (lambda x: True)
    method_condition = method_condition or (lambda x: True)
    Base.__getattribute__ = lambda self, name: _log_method_calls(  # type: ignore
        self, name, class_condition, method_condition
    )
    try:
        yield
    finally:
        Base.__getattribute__ = old  # type: ignore


def _repr_obj(obj, show_modules: bool = False, depth: int = 0) -> str:
    """Return a pretty representation of an object."""

    rep = f"{obj.__class__.__name__} ("
    if show_modules:
        rep = f"{obj.__class__.__module__}.{rep}"
    tab = "\t"

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
