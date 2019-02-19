from . import base


class Mode(base.RunningStatistic):
    """Compute online Mode.

    This class store in a dictionnary modalities and frequency of k first value of a given series.
    Mode allow to get modality value which have the higher frequency.
    If Mode set to exact, it will tore all the modalities of the target variable and return
    the exact mode, it could consume memory.

    Attributes:
        k (int): Number of modalities in the target variable.
        top (dic): Mapping of frequency of modalities.
        exact (bool): Store all the modalities of the target variable and returns the exact mode.

    Example:

        >>> from creme import stats

        >>> X = ['sunny', 'sunny', 'sunny', 'humidity', 'humidity', 'humidity', 'humidity']
        >>> mode = stats.mode.Mode(exact = True)
        >>> for x in X:
        ...     print(mode.update(x).get())
        sunny
        sunny
        sunny
        sunny
        sunny
        sunny
        humidity

        >>> mode = stats.mode.Mode(k=25, exact=False)
        >>> for x in X:
        ...     print(mode.update(x).get())
        sunny
        sunny
        sunny
        sunny
        sunny
        sunny
        humidity

    """

    def __init__(self, k=25, exact=False):
        self.k = k
        self.top = {}
        self.exact = exact

    @property
    def name(self):
        return 'mode'

    def update(self, x):
        if x in self.top:
            self.top[x] += 1
        elif len(self.top) <= self.k or self.exact:
            self.top[x] = 0
        return self

    def get(self):
        return max(self.top, key=self.top.get)
