import collections
import typing

class Rollable(typing.Protocol):

    def revert(self, *args, **kwargs):
        ...

class Rolling:

    def __init__(self, obj: Rollable, window_size: int):
        self.window: typing.Deque = collections.deque(maxlen=window_size)
        self.obj = obj

    def __getattribute__(self, name):
        try:
            return super().__getattribute__(name)
        except AttributeError:
            return self.obj.__getattribute__(name)

    def __getitem__(self, idx):
        return self.obj[idx]

    @property
    def window_size(self):
        return self.window.maxlen

    def update(self, *args, **kwargs):
        if len(self.window) == self.window_size:
            self.obj.revert(*self.window[0][0], **self.window[0][1])
        self.obj.update(*args, **kwargs)
        self.window.append((args, kwargs))
        return self

    def __repr__(self):
        return repr(self.obj)
