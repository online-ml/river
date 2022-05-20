import collections
import copy
import functools
import importlib
import inspect
import math
import pickle
import random
import statistics

import numpy as np
import pytest
from scipy import stats as sp_stats

from river import stats


def load_stats():
    for _, obj in inspect.getmembers(
        importlib.import_module("river.stats"), inspect.isclass
    ):
        try:

            if inspect.isabstract(obj):
                continue

            if issubclass(obj, stats.Link):
                yield obj(stats.Shift(1), stats.Mean())
                continue

            sig = inspect.signature(obj)
            yield obj(
                **{
                    param.name: param.default if param.default != param.empty else 1
                    for param in sig.parameters.values()
                }
            )
        except ValueError:
            yield obj()


@pytest.mark.parametrize("stat", load_stats(), ids=lambda stat: stat.__class__.__name__)
def test_pickling(stat):
    assert isinstance(pickle.loads(pickle.dumps(stat)), stat.__class__)
    assert isinstance(copy.deepcopy(stat), stat.__class__)

    # Check the statistic has a working __str__ and name method
    assert isinstance(str(stat), str)

    if isinstance(stat, stats.base.Univariate):
        assert isinstance(stat.name, str)


@pytest.mark.parametrize("stat", load_stats(), ids=lambda stat: stat.__class__.__name__)
def test_repr_with_no_updates(stat):
    assert isinstance(repr(stat), str)
    assert isinstance(str(stat), str)


@pytest.mark.parametrize(
    "stat, func",
    [
        (stats.Kurtosis(bias=True), sp_stats.kurtosis),
        (stats.Kurtosis(bias=False), functools.partial(sp_stats.kurtosis, bias=False)),
        (stats.Mean(), statistics.mean),
        (stats.Skew(bias=True), sp_stats.skew),
        (stats.Skew(bias=False), functools.partial(sp_stats.skew, bias=False)),
        (stats.Var(ddof=0), np.var),
        (stats.Var(), functools.partial(np.var, ddof=1)),
    ],
)
def test_univariate(stat, func):

    # Shut up
    np.warnings.filterwarnings("ignore")

    X = [random.random() for _ in range(30)]

    for i, x in enumerate(X):
        stat.update(x)
        if i >= 1:
            assert math.isclose(stat.get(), func(X[: i + 1]), abs_tol=1e-10)


@pytest.mark.parametrize(
    "stat, func",
    [
        (stats.RollingMean(3), statistics.mean),
        (stats.RollingMean(10), statistics.mean),
        (stats.RollingVar(3, ddof=0), np.var),
        (stats.RollingVar(10, ddof=0), np.var),
        (
            stats.RollingQuantile(0.0, 10),
            functools.partial(np.quantile, q=0.0, interpolation="linear"),
        ),
        (
            stats.RollingQuantile(0.25, 10),
            functools.partial(np.quantile, q=0.25, interpolation="linear"),
        ),
        (
            stats.RollingQuantile(0.5, 10),
            functools.partial(np.quantile, q=0.5, interpolation="linear"),
        ),
        (
            stats.RollingQuantile(0.75, 10),
            functools.partial(np.quantile, q=0.75, interpolation="linear"),
        ),
        (
            stats.RollingQuantile(1, 10),
            functools.partial(np.quantile, q=1, interpolation="linear"),
        ),
    ],
)
def test_rolling_univariate(stat, func):

    # We know what we're doing
    np.warnings.filterwarnings("ignore")

    def tail(iterable, n):
        return collections.deque(iterable, maxlen=n)

    n = stat.window_size
    X = [random.random() for _ in range(30)]

    for i, x in enumerate(X):
        stat.update(x)
        if i >= 1:
            assert math.isclose(stat.get(), func(tail(X[: i + 1], n)), abs_tol=1e-10)


@pytest.mark.parametrize(
    "stat, func",
    [
        (stats.Cov(), lambda x, y: np.cov(x, y)[0, 1]),
        (stats.PearsonCorr(), lambda x, y: sp_stats.pearsonr(x, y)[0]),
    ],
)
def test_bivariate(stat, func):

    # Shhh
    np.warnings.filterwarnings("ignore")

    X = [random.random() for _ in range(30)]
    Y = [random.random() * x for x in X]

    for i, (x, y) in enumerate(zip(X, Y)):
        stat.update(x, y)
        if i >= 1:
            assert math.isclose(stat.get(), func(X[: i + 1], Y[: i + 1]), abs_tol=1e-10)


@pytest.mark.parametrize(
    "stat, func",
    [
        (stats.RollingPearsonCorr(3), lambda x, y: sp_stats.pearsonr(x, y)[0]),
        (stats.RollingPearsonCorr(10), lambda x, y: sp_stats.pearsonr(x, y)[0]),
        (stats.RollingCov(3), lambda x, y: np.cov(x, y)[0, 1]),
        (stats.RollingCov(10), lambda x, y: np.cov(x, y)[0, 1]),
    ],
)
def test_rolling_bivariate(stat, func):

    # Enough already
    np.warnings.filterwarnings("ignore")

    def tail(iterable, n):
        return collections.deque(iterable, maxlen=n)

    n = stat.window_size
    X = [random.random() for _ in range(30)]
    Y = [random.random() * x for x in X]

    for i, (x, y) in enumerate(zip(X, Y)):
        stat.update(x, y)
        if i >= 1:
            x_tail = tail(X[: i + 1], n)
            y_tail = tail(Y[: i + 1], n)
            assert math.isclose(stat.get(), func(x_tail, y_tail), abs_tol=1e-10)


@pytest.mark.parametrize(
    "stat",
    filter(
        lambda stat: hasattr(stat, "update_many")
        and issubclass(stat.__class__, stats.base.Univariate),
        load_stats(),
    ),
    ids=lambda stat: stat.__class__.__name__,
)
def test_update_many_univariate(stat):

    batch_stat = stat.clone()

    for _ in range(5):
        X = np.random.random(10)
        batch_stat.update_many(X)
        for x in X:
            stat.update(x)

    assert math.isclose(batch_stat.get(), stat.get())


@pytest.mark.parametrize(
    "stat",
    filter(
        lambda stat: hasattr(stat, "update_many")
        and issubclass(stat.__class__, stats.base.Bivariate),
        load_stats(),
    ),
    ids=lambda stat: stat.__class__.__name__,
)
def test_update_many_bivariate(stat):

    batch_stat = stat.clone()

    for _ in range(5):
        X = np.random.random(10)
        Y = np.random.random(10)
        batch_stat.update_many(X, Y)
        for x, y in zip(X, Y):
            stat.update(x, y)

    assert math.isclose(batch_stat.get(), stat.get())
