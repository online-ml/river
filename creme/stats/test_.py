import collections
import copy
import functools
import importlib
import inspect
import math
import pickle
import random
import statistics

from creme import stats
import numpy as np
import pytest
from scipy import stats as sp_stats


def load_stats():
    for name, obj in inspect.getmembers(importlib.import_module('creme.stats'), inspect.isclass):
        try:
            sig = inspect.signature(obj)
            yield obj(**{
                param.name: param.default if param.default != param.empty else 1
                for param in sig.parameters.values()
            })
        except ValueError:
            yield obj()


@pytest.mark.parametrize('stat', load_stats(), ids=lambda stat: stat.__class__.__name__)
def test_pickling(stat):
    assert isinstance(pickle.loads(pickle.dumps(stat)), stat.__class__)
    assert isinstance(copy.deepcopy(stat), stat.__class__)

    # Check the statistic has a working __str__ and name method
    assert isinstance(str(stat), str)

    if isinstance(stat, stats.Univariate):
        assert isinstance(stat.name, str)


@pytest.mark.parametrize(
    'stat, func',
    [
        (stats.Kurtosis(bias=True), sp_stats.kurtosis),
        (stats.Kurtosis(bias=False), functools.partial(sp_stats.kurtosis, bias=False)),
        (stats.Mean(), statistics.mean),
        (stats.Skew(bias=True), sp_stats.skew),
        (stats.Skew(bias=False), functools.partial(sp_stats.skew, bias=False)),
        (stats.Var(ddof=0), np.var),
        (stats.Var(), functools.partial(np.var, ddof=1))
    ]
)
def test_univariate(stat, func):

    # Shut up
    np.warnings.filterwarnings('ignore')

    X = [random.random() for _ in range(30)]

    for i, x in enumerate(X):
        stat.update(x)
        try:
            assert math.isclose(stat.get(), func(X[:i + 1]), abs_tol=1e-10)
        except AssertionError:
            # Errors for the first value are acceptable
            if i == 0:
                continue


@pytest.mark.parametrize(
    'stat, func',
    [
        (stats.RollingMean(3), statistics.mean),
        (stats.RollingVar(3, ddof=0), np.var)
    ]
)
def test_rolling_univariate(stat, func):

    # We know what we're doing
    np.warnings.filterwarnings('ignore')

    def tail(iterable, n):
        return collections.deque(iterable, maxlen=n)

    n = stat.window_size
    X = [random.random() for _ in range(30)]

    for i, x in enumerate(X):
        stat.update(x)
        try:
            assert math.isclose(stat.get(), func(tail(X[:i + 1], n)), abs_tol=1e-10)
        except AssertionError:
            # Errors for the first value are acceptable
            if i == 0:
                continue


@pytest.mark.parametrize(
    'stat, func',
    [
        (stats.PearsonCorrelation(), lambda x, y: sp_stats.pearsonr(x, y)[0]),
    ]
)
def test_bivariate(stat, func):

    # Shhh
    np.warnings.filterwarnings('ignore')

    X = [random.random() for _ in range(30)]
    Y = [random.random() * x for x in X]

    for i, (x, y) in enumerate(zip(X, Y)):
        stat.update(x, y)
        if i >= 1:
            assert math.isclose(stat.get(), func(X[:i + 1], Y[:i + 1]), abs_tol=1e-10)
