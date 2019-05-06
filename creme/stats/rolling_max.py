from .. import utils

from . import base


class RollingMax(base.Univariate, utils.SortedWindow):
    """Running max over a window.

    Parameters:
        window_size (int): Size of the rolling window.

    Example:

        ::

            >>> from creme import stats

            >>> X = [1, -4, 3, -2, 2, 1]
            >>> rolling_max = stats.RollingMax(window_size=2)
            >>> for x in X:
            ...     print(rolling_max.update(x).get())
            1
            1
            3
            3
            2
            2

    """

    @property
    def name(self):
        return f'rolling_{self.window_size}_max'

    def update(self, x):
        self.append(x)
        return self

    def get(self):
        return self[-1]
