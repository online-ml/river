import collections

from . import _rolling_window


class RollingMode(_rolling_window._RollingWindow):
    """Compute windowed online Mode.

    Mode allow to get modality value which have the higher frequency with a given window size.

    Attributes:
        window_size (int): Size of the rolling window.
        top (defaultdic): Mapping of frequency of modalities.

    Example:

        >>> from creme import stats

        >>> X = ['sunny', 'sunny', 'sunny', 'humidity', 'humidity', 'humidity', 'humidity']
        >>> rolling_mode = stats.rolling_mode.RollingMode(window_size = 2)
        >>> for x in X:
        ...     print(rolling_mode.update(x).get())
        sunny
        sunny
        sunny
        sunny
        humidity
        humidity
        humidity

        >>> rolling_mode = stats.rolling_mode.RollingMode(window_size = 5)
        >>> for x in X:
        ...     print(rolling_mode.update(x).get())
        sunny
        sunny
        sunny
        sunny
        sunny
        humidity
        humidity

    """

    def __init__(self, window_size):
        super().__init__(window_size)
        self.top = collections.defaultdict(int)

    @property
    def name(self):
        return 'rolling_mode'

    def update(self, x):
        if len(self.rolling_window) >= self.window_size:
            self.top[self.rolling_window[0]] -= 1
            if self.top[self.rolling_window[0]] == 0:
                self.top.pop(self.rolling_window[0])

        self.top[x] += 1
        super().append(x)
        return self

    def get(self):
        return max(self.top, key=self.top.get)
