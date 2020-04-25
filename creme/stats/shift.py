import collections
import typing

from . import base

__all__ = ['Shift']


class Shift(base.Univariate):
    """Shift the observations with the selected period.

    Create a lag of `period` observations in the data stream. The `update` method allows
    to get the input of the shift statistic observed `period` ago. The Shift class
    can be combined with other statistics to create features with lag.

    Parameters:
        period: Number of periods to shift. Must be positive.
        missing: Handle missing values when there is'nt any reference for a given period.
        fit_before_transform: Shift logic. Must be set to True when the user updates the
            buffer before retrieving the observation with the offset selected.

    Example:

        ::

            >>> import creme

            >>> shift = creme.stats.Shift(period=2)

            >>> X = [
            ...    1,
            ...    2,
            ...    3,
            ...    4,
            ...    5,
            ... ]

            Shift a serie of values:
            >>> for x in X:
            ...    print(shift.update(x).get())
            0.
            0.
            1
            2
            3

            Compute shifted statistics:
            >>> statistics = creme.stats.Shift(2) | creme.stats.RollingMean(1)

            >>> for x in X:
            ...    print(statistics.update(x).get())
            0.
            0.
            1.
            2.
            3.

            You can include Shift into a pipeline such as:
            >>> import creme

            >>> X = [
            ...     {'place': 'Taco Bell', 'revenue': 42},
            ...     {'place': 'Burger King', 'revenue': 16},
            ...     {'place': 'Burger King', 'revenue': 24},
            ...     {'place': 'Taco Bell', 'revenue': 58},
            ...     {'place': 'Burger King', 'revenue': 20},
            ...     {'place': 'Taco Bell', 'revenue': 50}
            ... ]

            >>> agg = creme.feature_extraction.TargetAgg(
            ...     by='place',
            ...     how=creme.stats.Shift(1, missing=25) | creme.stats.Mean()
            ... )

            >>> for x in X:
            ...     print(agg.transform_one(x))
            ...     y = x.pop('revenue')
            ...     agg = agg.fit_one(x, y)
            {'target_shift_1_mean_by_place': 0.0}
            {'target_shift_1_mean_by_place': 0.0}
            {'target_shift_1_mean_by_place': 25.0}
            {'target_shift_1_mean_by_place': 25.0}
            {'target_shift_1_mean_by_place': 20.5}
            {'target_shift_1_mean_by_place': 33.5}

    """
    def __init__(self, period: int, missing: float=0., fit_before_transform: bool=True):
        self.period = period
        self.missing = missing
        self.period_logic = period
        if fit_before_transform:
            self.period += 1
        self.buffer = collections.deque(maxlen=self.period)

    @property
    def name(self):
        return f'shift_{self.period_logic}'

    def update(self, x):
        self.buffer.append(x)
        return self

    def get(self):
        if len(self.buffer) == self.period:
            return self.buffer[0]
        return self.missing
