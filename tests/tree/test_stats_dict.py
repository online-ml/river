from __future__ import annotations

import collections

import pytest

from river import stats
from river.tree.utils import combine_stats, update_stats


def make_var(*values):
    result = stats.Var()
    for value in values:
        result.update(value)
    return result


def assert_var_equal(left, right):
    assert left.n == right.n
    assert left.mean.get() == pytest.approx(right.mean.get())
    assert left.get() == pytest.approx(right.get())


def test_default_value():
    values = collections.defaultdict(stats.Var)
    values["a"].update(1.0)
    assert values["a"].get() == 0.0


def test_addition_and_subtraction():
    left = {"a": make_var(1.0, 2.0), "b": make_var(3.0, 4.0)}
    right = {"b": make_var(5.0, 6.0), "c": make_var(7.0, 8.0)}

    combined = combine_stats(left, right)
    assert_var_equal(combined["a"], left["a"])
    assert_var_equal(combined["b"], make_var(3.0, 4.0, 5.0, 6.0))
    assert_var_equal(combined["c"], right["c"])

    restored = combine_stats(combined, right, subtract=True)
    assert_var_equal(restored["a"], left["a"])
    assert_var_equal(restored["b"], left["b"])
    assert restored["c"].n == 0.0


def test_in_place_operations_do_not_alias_new_values():
    left = {}
    right = {"a": make_var(1.0, 2.0)}

    update_stats(left, right)
    left["a"].update(3.0)
    assert right["a"].n == 2.0

    update_stats(left, right, subtract=True)
    assert left["a"].n == 1.0
