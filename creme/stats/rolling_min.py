from .. import utils

from . import base


class RollingMin(base.Univariate, utils.SortedWindow):
    """Running min over a window.

    Parameters:
        window_size (int): Size of the rolling window.

    Example:

        ::

            >>> from creme import stats

            >>> X = [1, -4, 3, -2, 2, 1]
            >>> rolling_min = stats.RollingMin(2)
            >>> for x in X:
            ...     print(rolling_min.update(x).get())
            1
            -4
            -4
            -2
            -2
            1

    """

    @property
    def name(self):
        return f'rolling_{self.window_size}_min'

    def update(self, x):
        self.append(x)
        return self

    def get(self):
        return self[0]
