import bisect
import datetime as dt
import typing

from . import base


__all__ = ["TimeRolling"]


class TimeRolling(base.WrapperMetric):
    """Wrapper for computing metrics over a period of time.

    Parameters
    ----------
    metric
        A metric.
    period
        A period of time.

    Examples
    --------

    >>> import datetime as dt
    >>> from river import metrics

    >>> y_true = [3, -0.5, 2, 7]
    >>> y_pred = [2.5, 0.0, 2, 9]
    >>> days = [1, 2, 3, 4]

    >>> metric = metrics.TimeRolling(metrics.MAE(), period=dt.timedelta(days=2))

    >>> for yt, yp, day in zip(y_true, y_pred, days):
    ...     t = dt.datetime(2019, 1, day)
    ...     print(metric.update(yt, yp, t))
    MAE: 0.5
    MAE: 0.5
    MAE: 0.25
    MAE: 1.

    """

    def __init__(self, metric: base.Metric, period: dt.timedelta):
        self._metric = metric
        self.period = period
        self._events: typing.List[typing.Tuple[dt.datetime, typing.Any, typing.Any]] = []
        self._latest = dt.datetime(1, 1, 1)

    @property
    def metric(self):
        return self._metric

    def update(self, y_true, y_pred, t):
        self.metric.update(y_true, y_pred)
        bisect.insort_left(self._events, (t, y_true, y_pred))

        # There will only be events to revert if the new event if younger than the previously seen
        # youngest event
        if t > self._latest:
            self._latest = t

            i = 0
            for ti, yt, yp in self._events:
                if ti > t - self.period:
                    break
                self.metric.revert(yt, yp)
                i += 1

            # Remove expired events
            if i > 0:
                self._events = self._events[i:]

        return self

    def revert(self, y_true, y_pred):
        raise NotImplementedError
