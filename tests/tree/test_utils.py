from __future__ import annotations

import sys

import pytest

from river import tree
from river.datasets import synth
from river.tree.utils import calculate_object_size


class SlottedState:
    __slots__ = ("value",)


class InheritedState(SlottedState):
    __slots__ = ("child", "unset")


def test_calculate_object_size_splitter_growth():
    splitter = tree.splitter.ExhaustiveSplitter()
    splitter.update(0, 0, 1)
    initial_size = calculate_object_size(splitter)

    for i in range(1, 100):
        splitter.update(i, i % 2, 1)

    assert calculate_object_size(splitter) > initial_size + 99 * sys.getsizeof(splitter._root)


def test_calculate_object_size_nested_inherited_slots():
    state = InheritedState()
    state.value = bytearray(1024)
    state.child = SlottedState()
    state.child.value = bytearray(2048)

    expected = sum(
        sys.getsizeof(obj) for obj in (state, state.value, state.child, state.child.value)
    )
    assert calculate_object_size(state) == expected


def test_calculate_object_size_shared_cycles():
    state = InheritedState()
    state.value = bytearray(1024)
    state.child = state
    objects = [state, state, state.value]

    expected = sum(sys.getsizeof(obj) for obj in (objects, state, state.value))
    assert calculate_object_size(objects) == expected


def test_calculate_object_size_preserves_generator():
    values = (i for i in range(3))

    calculate_object_size(values)

    assert list(values) == [0, 1, 2]


@pytest.mark.parametrize(
    "unit, divisor", [("byte", 1), ("KiB", 1024), ("MiB", 2**20), ("other", 1)]
)
def test_calculate_object_size_units(unit, divisor):
    state = bytearray(2048)

    size = calculate_object_size(state, unit)

    assert isinstance(size, int)
    assert size == int(sys.getsizeof(state) / divisor)


def test_memory_estimate_deactivates_efdt_leaves():
    model = tree.ExtremelyFastDecisionTreeClassifier(
        leaf_prediction="mc",
        max_size=0.05,
        grace_period=50,
        memory_estimate_period=50,
        splitter=tree.splitter.ExhaustiveSplitter(),
    )
    deactivated = False

    for x, y in synth.LED(seed=42).take(500):
        model.learn_one(x, y)
        deactivated |= model.n_inactive_leaves > 0

    assert deactivated
    assert model.n_active_leaves == sum(leaf.is_active() for leaf in model._find_leaves())
    assert model.n_inactive_leaves == sum(not leaf.is_active() for leaf in model._find_leaves())
    assert calculate_object_size(model) == model._raw_memory_usage
    assert model._raw_memory_usage / (2**20) < 0.066
