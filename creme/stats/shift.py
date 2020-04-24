import collections
import typing

from . import base

__all__ = ['Shift']


class Shift(base.Univariate):
    """Shift the observations with the selected period.

    Shift class allows to store and recall at regular intervals values observed in a stream.
    The method `get()` returns the n-th - period observed entry.

    Parameters:
        period: Number of periods to shift. Must be positive.
        missing: Handle missing values when there is'nt any reference for a given period.
        fit_before_transform: Shift logic. Must be set to True when the user updates the
            buffer before retrieving the observation with the offset selected.

    Example:

        ::

            >>> import creme

            >>> shift = creme.stats.Shift(period=2, missing=-1)

            >>> X = [
            ...    1,
            ...    2,
            ...    3,
            ...    4,
            ...    5,
            ... ]

            >>> for x in X:
            ...    print(shift.update(x).get())
            -1
            -1
            1
            2
            3

            It is possible to compute shifted statistics with operator |.

            >>> statistics = creme.stats.Shift(2) | creme.stats.RollingMean(1)
            >>> for x in X:
            ...    print(statistics.update(x).get())
            0.0
            0.0
            1.0
            2.0
            3.0

            You can include Shift into a pipeline such as:

            >>> X_y = [
            ...    ({'store_id': 'darty', 'open': 1}, 1.),
            ...    ({'store_id': 'fnac', 'open': 0}, 2.),
            ...    ({'store_id': 'darty', 'open': 3}, 3.),
            ...    ({'store_id': 'darty', 'open': 1}, 4.),
            ...    ({'store_id': 'ikea', 'open': 1}, 5.),
            ...    ({'store_id': 'ikea', 'open': 1}, 10.),
            ... ]

            >>> pipeline = creme.feature_extraction.Agg(
            ...         by=['store_id'],
            ...         on='open',
            ...         how=creme.stats.Shift(1) | creme.stats.Sum()
            ...     ) + creme.feature_extraction.TargetAgg(
            ...             by=['store_id'],
            ...             how=creme.stats.Shift(1) | creme.stats.RollingMean(1)
            ... )

            >>> for x, y in X_y:
            ...  print(pipeline.fit_one(x, y).transform_one(x))
            {'target_shift_1_rolling_rollingmean_1_by_store_id': 0.0, 'open_shift_1_sum_by_store_id': 0.0}
            {'target_shift_1_rolling_rollingmean_1_by_store_id': 0.0, 'open_shift_1_sum_by_store_id': 0.0}
            {'target_shift_1_rolling_rollingmean_1_by_store_id': 1.0, 'open_shift_1_sum_by_store_id': 1.0}
            {'target_shift_1_rolling_rollingmean_1_by_store_id': 3.0, 'open_shift_1_sum_by_store_id': 4.0}
            {'target_shift_1_rolling_rollingmean_1_by_store_id': 0.0, 'open_shift_1_sum_by_store_id': 0.0}
            {'target_shift_1_rolling_rollingmean_1_by_store_id': 5.0, 'open_shift_1_sum_by_store_id': 1.0}

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
