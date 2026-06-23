from __future__ import annotations

import bisect
import collections
import datetime as dt
import inspect
import warnings
from typing import Any, Generic, Protocol, TypeVar, cast, overload, runtime_checkable


@runtime_checkable
class Rollable(Protocol):
    def update(self, *args: Any, **kwargs: Any) -> None: ...

    def revert(self, *args: Any, **kwargs: Any) -> None: ...


_T = TypeVar("_T", bound=Rollable)


_INSTANCE_DEPRECATION = (
    "Passing an instance to utils.{cls} is deprecated and will be removed in a future release. "
    "Pass the class and forward constructor kwargs instead, "
    "e.g. utils.{cls}(stats.Mean, {window_kw}=...) "
    "or utils.{cls}(stats.Var, {window_kw}=..., ddof=0)."
)


class BaseRolling(Generic[_T]):
    def __init__(self, obj: _T) -> None:
        self.obj = obj

    def __getattr__(self, name: str) -> object:
        # Only called when normal attribute lookup fails, so the fast path
        # (self.obj, self.window, etc.) never enters this method.
        # Guard against recursion during deepcopy/pickle when obj is not yet set.
        try:
            obj = object.__getattribute__(self, "obj")
        except AttributeError:
            raise AttributeError(name)
        return getattr(obj, name)

    def __getitem__(self, idx: Any) -> object:
        # Enable for when it needs, throws a runtime error as usual if tried on a type that can't.
        return self.obj[idx]  # type: ignore[index]

    def __repr__(self) -> str:
        return repr(self.obj)


class Rolling(BaseRolling[_T]):
    """A generic wrapper for performing rolling computations.

    This can be wrapped around any class whose instances implement both an ``update`` and a
    ``revert`` method. The wrapped class is instantiated internally; any extra keyword
    arguments are forwarded to its constructor. Inputs to ``update`` are stored in a queue,
    and elements of the queue are popped when the window is full.

    Parameters
    ----------
    cls
        A class whose instances implement ``update`` and ``revert``. Passing a pre-built
        instance is deprecated and will be removed in a future release.
    window_size
        Size of the window.
    **kwargs
        Forwarded to ``cls`` when instantiating the wrapped object.

    Examples
    --------

    For instance, here is how you can compute a rolling average over a window of size 3:

    >>> from river import stats, utils

    >>> X = [1, 3, 5, 7]
    >>> rmean = utils.Rolling(stats.Mean, window_size=3)

    >>> for x in X:
    ...     rmean.update(x)
    ...     print(rmean.get())
    1.0
    2.0
    3.0
    5.0

    Constructor arguments for the wrapped class are passed as keyword arguments:

    >>> rvar = utils.Rolling(stats.Var, window_size=3, ddof=0)

    """

    @overload
    def __init__(self, cls: type[_T], window_size: int, **kwargs: Any) -> None: ...
    @overload
    def __init__(self, cls: _T, window_size: int) -> None: ...
    def __init__(self, cls: type[_T] | _T, window_size: int, **kwargs: Any) -> None:
        if inspect.isclass(cls):
            obj = cls(**kwargs)
        else:
            if kwargs:
                raise TypeError(
                    "utils.Rolling received constructor kwargs alongside a pre-built instance. "
                    "Pass the class instead, e.g. utils.Rolling(stats.Var, window_size=..., ddof=0)."
                )
            warnings.warn(
                _INSTANCE_DEPRECATION.format(cls="Rolling", window_kw="window_size"),
                DeprecationWarning,
                stacklevel=2,
            )
            obj = cls
        if not isinstance(obj, Rollable):
            raise ValueError(f"{obj} does not satisfy the necessary protocol")
        super().__init__(cast(_T, obj))
        self._window_size = window_size
        self.window: collections.deque[tuple[tuple[Any, ...], dict[str, Any]]] = collections.deque(
            maxlen=window_size
        )

    @property
    def window_size(self) -> int:
        return self._window_size

    def update(self, *args: Any, **kwargs: Any) -> None:
        window = self.window
        if len(window) == self._window_size:
            old = window[0]
            self.obj.revert(*old[0], **old[1])
        self.obj.update(*args, **kwargs)
        window.append((args, kwargs))


class TimeRolling(BaseRolling[_T]):
    """A generic wrapper for performing time rolling computations.

    This can be wrapped around any class whose instances implement both an ``update`` and a
    ``revert`` method. The wrapped class is instantiated internally; any extra keyword
    arguments are forwarded to its constructor. Inputs to ``update`` are stored in a queue,
    and elements of the queue are popped when they are too old.

    Parameters
    ----------
    cls
        A class whose instances implement ``update`` and ``revert``. Passing a pre-built
        instance is deprecated and will be removed in a future release.
    period
        A duration of time, expressed as a `datetime.timedelta`.
    **kwargs
        Forwarded to ``cls`` when instantiating the wrapped object.

    Examples
    --------

    For instance, here is how you can compute a rolling average over a period of 3 days:

    >>> from river import stats, utils

    >>> X = {
    ...     dt.datetime(2019, 1, 1): 1,
    ...     dt.datetime(2019, 1, 2): 5,
    ...     dt.datetime(2019, 1, 3): 9,
    ...     dt.datetime(2019, 1, 4): 13
    ... }

    >>> rmean = utils.TimeRolling(stats.Mean, period=dt.timedelta(days=3))
    >>> for t, x in X.items():
    ...     rmean.update(x, t=t)
    ...     print(rmean.get())
    1.0
    3.0
    5.0
    9.0

    """

    @overload
    def __init__(self, cls: type[_T], period: dt.timedelta, **kwargs: Any) -> None: ...
    @overload
    def __init__(self, cls: _T, period: dt.timedelta) -> None: ...
    def __init__(self, cls: type[_T] | _T, period: dt.timedelta, **kwargs: Any) -> None:
        if inspect.isclass(cls):
            obj = cls(**kwargs)
        else:
            if kwargs:
                raise TypeError(
                    "utils.TimeRolling received constructor kwargs alongside a pre-built instance. "
                    "Pass the class instead, e.g. utils.TimeRolling(stats.Var, period=..., ddof=0)."
                )
            warnings.warn(
                _INSTANCE_DEPRECATION.format(cls="TimeRolling", window_kw="period"),
                DeprecationWarning,
                stacklevel=2,
            )
            obj = cls
        if not isinstance(obj, Rollable):
            raise ValueError(f"{obj} does not satisfy the necessary protocol")
        super().__init__(cast(_T, obj))
        self.period = period
        self._timestamps: list[dt.datetime] = []
        self._datum: list[Any] = []
        self._latest = dt.datetime(1, 1, 1)

    def update(self, *args: Any, t: dt.datetime, **kwargs: Any) -> None:
        self.obj.update(*args, **kwargs)
        i = bisect.bisect_left(self._timestamps, t)
        self._timestamps.insert(i, t)
        self._datum.insert(i, (args, kwargs))

        # There will only be events to revert if the new event if younger than the previously seen
        # youngest event
        if t > self._latest:
            self._latest = t

            i = 0
            for ti, (argsi, kwargsi) in zip(self._timestamps, self._datum):
                if ti > t - self.period:
                    break
                self.obj.revert(*argsi, **kwargsi)
                i += 1

            # Remove expired events
            if i > 0:
                self._timestamps = self._timestamps[i:]
                self._datum = self._datum[i:]
