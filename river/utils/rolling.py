from __future__ import annotations

import bisect
import collections
import datetime as dt
import typing


@typing.runtime_checkable
class Rollable(typing.Protocol):
    def update(self, *args, **kwargs):
        ...

    def revert(self, *args, **kwargs):
        ...


class BaseRolling:
    def __init__(self, obj: Rollable):
        if not isinstance(obj, Rollable):
            raise ValueError(f"{obj} does not satisfy the necessary protocol")

        self.obj = obj

    def __getattribute__(self, name):
        try:
            return super().__getattribute__(name)
        except AttributeError:
            return super().__getattribute__("obj").__getattribute__(name)

    def __getitem__(self, idx):
        return self.obj[idx]

    def __repr__(self):
        return repr(self.obj)


class Rolling(BaseRolling):
    """A generic wrapper for performing rolling computations.

    This can be wrapped around any object which implements both an `update` and a `revert` method.
    Inputs to `update` are stored in a queue. Elements of the queue are popped when the window is
    full.

    Parameters
    ----------
    obj
        An object that implements both an `update` method and a `rolling `method.
    window_size
        Size of the window.

    Examples
    --------

    For instance, here is how you can compute a rolling average over a window of size 3:

    >>> from river import stats, utils

    >>> X = [1, 3, 5, 7]
    >>> rmean = utils.Rolling(stats.Mean(), window_size=3)

    >>> for x in X:
    ...     print(rmean.update(x).get())
    1.0
    2.0
    3.0
    5.0

    """

    def __init__(self, obj: Rollable, window_size: int):
        super().__init__(obj)
        self.window: collections.deque = collections.deque(maxlen=window_size)

    @property
    def window_size(self):
        return self.window.maxlen

    def update(self, *args, **kwargs):
        if len(self.window) == self.window_size:
            self.obj.revert(*self.window[0][0], **self.window[0][1])
        self.obj.update(*args, **kwargs)
        self.window.append((args, kwargs))
        return self


class TimeRolling(BaseRolling):
    """A generic wrapper for performing time rolling computations.

    This can be wrapped around any object which implements both an `update` and a `revert` method.
    Inputs to `update` are stored in a queue. Elements of the queue are popped when they are too
    old.

    Parameters
    ----------
    obj
        An object that implements both an `update` method and a `rolling `method.
    period
        A duration of time, expressed as a `datetime.timedelta`.

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

    >>> rmean = utils.TimeRolling(stats.Mean(), period=dt.timedelta(days=3))
    >>> for t, x in X.items():
    ...     print(rmean.update(x, t=t).get())
    1.0
    3.0
    5.0
    9.0

    """

    def __init__(self, obj: Rollable, period: dt.timedelta):
        super().__init__(obj)
        self.period = period
        self._events: list[tuple[dt.datetime, typing.Any]] = []
        self._latest = dt.datetime(1, 1, 1)

    def update(self, *args, t: dt.datetime, **kwargs):
        self.obj.update(*args, **kwargs)
        bisect.insort_left(self._events, (t, (args, kwargs)))

        # There will only be events to revert if the new event if younger than the previously seen
        # youngest event
        if t > self._latest:
            self._latest = t

            i = 0
            for ti, (argsi, kwargsi) in self._events:
                if ti > t - self.period:
                    break
                self.obj.revert(*argsi, **kwargsi)
                i += 1

            # Remove expired events
            if i > 0:
                self._events = self._events[i:]

        return self
