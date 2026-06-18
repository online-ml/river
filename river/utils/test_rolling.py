from __future__ import annotations

import datetime as dt

import pytest

from river import proba, stats, utils


def test_with_counter() -> None:
    """
    >>> from river import utils
    >>> import collections
    >>> collections.Counter.revert = collections.Counter.subtract

    >>> counter = utils.Rolling(collections.Counter, window_size=3)

    >>> for i in range(5):
    ...     counter.update([i])

    >>> counter
    Counter({2: 1, 3: 1, 4: 1, 0: 0, 1: 0})

    >>> counter.most_common(3)
    [(2, 1), (3, 1), (4, 1)]

    >>> counter[4]
    1

    """


def test_rolling_with_not_rollable() -> None:
    with pytest.raises(ValueError):
        utils.Rolling(stats.Quantile, window_size=10)  # type: ignore[type-var]


def test_time_rolling_with_not_rollable() -> None:
    with pytest.raises(ValueError):
        utils.TimeRolling(stats.Quantile, period=dt.timedelta(seconds=10))  # type: ignore[type-var]


def test_rolling_forwards_kwargs_to_class() -> None:
    rvar = utils.Rolling(stats.Var, window_size=3, ddof=0)
    for x in [1, 2, 3]:
        rvar.update(x)
    assert rvar.get() == pytest.approx(2 / 3)


def test_rolling_instance_is_deprecated() -> None:
    with pytest.warns(DeprecationWarning, match="deprecated"):
        utils.Rolling(stats.Mean(), window_size=3)


def test_time_rolling_instance_is_deprecated() -> None:
    with pytest.warns(DeprecationWarning, match="deprecated"):
        utils.TimeRolling(stats.Mean(), period=dt.timedelta(seconds=10))


def test_rolling_rejects_kwargs_alongside_instance() -> None:
    with pytest.raises(TypeError, match="instance"):
        utils.Rolling(stats.Var(), window_size=3, ddof=0)  # type: ignore[call-overload]


def test_issue_1343() -> None:
    """

    https://github.com/online-ml/river/issues/1343

    """
    rmean = utils.TimeRolling(proba.MultivariateGaussian, period=dt.timedelta(microseconds=1))
    t = dt.datetime.now()
    rmean.update({"a": 0}, t=t)
    rmean.update({"a": 1}, t=t)
