import collections
import copy
import inspect
import sys
import types
import typing


class Base:
    """Base class that is inherited by the majority of classes in River.

    This base class allows us to handle the following tasks in a uniform manner:

    - Getting and setting parameters.
    - Displaying information.
    - Cloning.

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
        return {}

    def _get_params(self) -> typing.Dict[str, typing.Any]:
        """Return the parameters that were used during initialization."""

        params = {}

        for name, param in inspect.signature(self.__init__).parameters.items():

            # *args
            if param.kind == param.VAR_POSITIONAL:
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

    def _set_params(self, new_params: dict = None):
        """Return a new instance with the current parameters as well as new ones.

        Calling this without any parameters will essentially clone the estimator.

        Parameters
        ----------
        new_params

        Examples
        --------

        >>> from river import linear_model
        >>> from river import optim

        >>> model = linear_model.LinearRegression(
        ...     optimizer=optim.SGD(lr=0.042),
        ... )

        >>> new_params = {
        ...     'optimizer': (optim.SGD, {'lr': .001})
        ... }

        >>> model._set_params(new_params)
        LinearRegression (
          optimizer=SGD (
            lr=Constant (
              learning_rate=0.001
            )
          )
          loss=Squared ()
          l2=0.
          intercept_init=0.
          intercept_lr=Constant (
            learning_rate=0.01
          )
          clip_gradient=1e+12
          initializer=Zeros ()
        )

        The algorithm will be recursively called down `Pipeline`s and `TransformerUnion`s.

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
        ...         'optimizer': (optim.SGD, {'lr': .03})
        ...     }
        ... }

        >>> model._set_params(new_params)
        Pipeline (
          StandardScaler (),
          LinearRegression (
            optimizer=SGD (
              lr=Constant (
                learning_rate=0.03
              )
            )
            loss=Squared ()
            l2=0.
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
            return (
                isinstance(param, tuple)
                and inspect.isclass(param[0])
                and isinstance(param[1], dict)
            )

        def instantiate(klass, params, new_params):

            params = {
                name: new_params.get(name, param) for name, param in params.items()
            }

            return klass(
                **{
                    name: (
                        instantiate(klass=param[0], params=param[1], new_params={})
                        if is_class_param(param)
                        else copy.deepcopy(param)
                    )
                    for name, param in params.items()
                }
            )

        if new_params is None:
            new_params = {}

        return instantiate(self.__class__, self._get_params(), new_params)

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
            elif hasattr(obj, "__iter__") and not isinstance(
                obj, (str, bytes, bytearray)
            ):
                buffer.extend([i for i in obj])  # noqa

        return size

    @property
    def _memory_usage(self) -> str:
        """Return the memory usage in a human readable format."""
        from river import utils

        return utils.pretty.humanize_bytes(self._raw_memory_usage)


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
