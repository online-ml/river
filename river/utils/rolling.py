import bisect
import collections
import datetime as dt
import typing


class Rollable(typing.Protocol):
    def update(self, *args, **kwargs):
        ...

    def revert(self, *args, **kwargs):
        ...


class BaseRolling:
    def __init__(self, obj: Rollable):
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
    def __init__(self, obj: Rollable, window_size: int):
        super().__init__(obj)
        self.window: typing.Deque = collections.deque(maxlen=window_size)

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
    def __init__(self, obj: Rollable, period: dt.timedelta):
        super().__init__(obj)
        self.period = period
        self._events: typing.List[typing.Tuple[dt.datetime, typing.Any]] = []
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
