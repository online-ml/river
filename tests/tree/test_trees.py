from __future__ import annotations

import random

import pytest

from river import drift, tree
from river.datasets import synth


def get_classification_data():
    return synth.LED(seed=42).take(500)


def get_regression_data():
    return synth.Friedman(seed=42).take(500)


@pytest.mark.parametrize(
    "dataset, model",
    [
        (
            get_classification_data(),
            tree.HoeffdingTreeClassifier(
                leaf_prediction="mc",
                max_size=0.05,
                grace_period=50,
                memory_estimate_period=50,
                splitter=tree.splitter.ExhaustiveSplitter(),
            ),
        ),
        (
            get_classification_data(),
            tree.HoeffdingAdaptiveTreeClassifier(
                leaf_prediction="mc",
                max_size=0.05,
                grace_period=50,
                memory_estimate_period=50,
                splitter=tree.splitter.ExhaustiveSplitter(),
            ),
        ),
        (
            get_classification_data(),
            tree.ExtremelyFastDecisionTreeClassifier(
                leaf_prediction="mc",
                max_size=0.05,
                grace_period=50,
                memory_estimate_period=50,
                splitter=tree.splitter.ExhaustiveSplitter(),
            ),
        ),
    ],
)
def test_memory_usage_class(dataset, model):
    for x, y in dataset:
        model.learn_one(x, y)

    assert model._raw_memory_usage / (2**20) < 0.066


@pytest.mark.parametrize(
    "dataset, model",
    [
        (
            get_regression_data(),
            tree.HoeffdingTreeRegressor(
                leaf_prediction="mean", max_size=0.5, memory_estimate_period=100
            ),
        ),
        (
            get_regression_data(),
            tree.HoeffdingAdaptiveTreeRegressor(
                leaf_prediction="mean", max_size=0.5, memory_estimate_period=100
            ),
        ),
    ],
)
def test_memory_usage_reg(dataset, model):
    for x, y in dataset:
        model.learn_one(x, y)

    assert model._raw_memory_usage / (2**20) < 0.5


def test_memory_usage_multitarget():
    dataset = get_regression_data()

    model = tree.ISOUPTreeRegressor(
        leaf_prediction="mean",
        max_size=0.5,
        memory_estimate_period=100,
    )

    for x, y in dataset:
        # Handcrafted targets
        y_ = {0: y, 1: 2 * y, 2: 3 * y}
        model.learn_one(x, y_)

    assert model._raw_memory_usage / (2**20) < 0.5


def test_efdt_split_reevaluation():
    dataset = synth.SEA(seed=7, variant=2).take(500)

    model = tree.ExtremelyFastDecisionTreeClassifier(
        leaf_prediction="nb",
        grace_period=50,
        min_samples_reevaluate=10,
        split_criterion="hellinger",
        delta=0.1,
    )

    max_depth = -1
    for x, y in dataset:
        model.learn_one(x, y)

        if model.height > max_depth:
            max_depth = model.height

    assert model.height != max_depth


@pytest.mark.parametrize("depth", [0, 1], ids=["root", "internal"])
@pytest.mark.parametrize(
    "model_class",
    [tree.HoeffdingAdaptiveTreeClassifier, tree.ExtremelyFastHoeffdingAdaptiveTreeClassifier],
)
def test_hatc_alternate_split_stays_with_owner(depth, model_class):
    model = model_class(
        grace_period=8,
        delta=0.9,
        tau=0,
        bootstrap_sampling=False,
        leaf_prediction="mc",
        drift_detector=drift.NoDrift(),
        drift_window_threshold=1000,
    )
    for i in range(8):
        model.learn_one({"side": str(i % 2)}, i % 2)
    for i in range(8):
        model.learn_one({"side": "1", "signal": str(i % 2)}, i % 2)

    x = {"side": "1", "signal": "0"}
    root = model._root
    owner = root if depth == 0 else root.next(x)
    assert owner.depth == depth
    owner.drift_detector = drift.DummyDriftDetector(t_0=1)
    model.learn_one(x, 1 - model.predict_one(x))
    assert owner._alternate_tree is not None
    owner.drift_detector = drift.NoDrift()
    for i in range(1, 8):
        model.learn_one({"side": "1", "signal": str(i % 2)}, i % 2)

    assert model.n_switch_alternate_trees == 0
    assert model._root is root
    assert (root if depth == 0 else root.next(x)) is owner
    assert owner._alternate_tree.n_branches == 1
    assert model.n_active_leaves == model.n_leaves + owner._alternate_tree.n_leaves
    assert sum(model.predict_proba_one(x).values()) == pytest.approx(1)


@pytest.mark.parametrize(
    ("model_class", "expected_branches"),
    [
        (tree.HoeffdingAdaptiveTreeClassifier, 0),
        (tree.ExtremelyFastHoeffdingAdaptiveTreeClassifier, 1),
    ],
)
@pytest.mark.parametrize("criterion", ["info_gain", "gini", "hellinger"])
def test_tied_split_candidates(model_class, expected_branches, criterion):
    model = model_class(
        grace_period=20, delta=0.1, tau=0.0, bootstrap_sampling=False, split_criterion=criterion
    )
    for i in range(20):
        y = i % 2
        value = str(1 - y if i < 2 else y)
        model.learn_one({"a": value, "b": value}, y)
    assert model.n_branches == expected_branches


@pytest.mark.parametrize("criterion", ["info_gain", "gini", "hellinger"])
def test_efhat_single_uninformative_candidate(criterion):
    model = tree.ExtremelyFastHoeffdingAdaptiveTreeClassifier(
        grace_period=20,
        tau=0.0,
        bootstrap_sampling=False,
        merit_preprune=False,
        split_criterion=criterion,
    )
    for i in range(20):
        model.learn_one({"a": str(i % 2)}, (i // 2) % 2)
    assert model.n_branches == 0


def test_drift_adaptation_hatc():
    rng = random.Random(42)
    dataset = iter(synth.Sine(seed=8, classification_function=0, has_noise=True))

    model = tree.HoeffdingAdaptiveTreeClassifier(
        leaf_prediction="mc",
        grace_period=10,
        drift_detector=drift.ADWIN(0.1),
        delta=0.1,
        drift_window_threshold=2,
        seed=42,
        max_depth=3,
    )

    for i in range(1000):
        if i % 200 == 0 and i > 0:
            dataset = iter(
                synth.Sine(seed=8, classification_function=rng.randint(0, 3), has_noise=False)
            )

        x, y = next(dataset)
        model.learn_one(x, y)

    assert model._n_switch_alternate_trees > 0


def test_drift_adaptation_hatr():
    dataset = synth.Friedman(seed=7).take(1000)

    model = tree.HoeffdingAdaptiveTreeRegressor(
        leaf_prediction="mean",
        grace_period=10,
        delta=0.1,
        drift_detector=drift.ADWIN(0.1),
        drift_window_threshold=10,
        seed=7,
        max_depth=3,
    )

    for i, (x, y) in enumerate(dataset):
        y_ = y
        if i > 500:
            # Emulate an abrupt drift
            y_ = 1.5 * y
        model.learn_one(x, y_)

    assert model._n_alternate_trees > 0
