import random

import pytest

from river import datasets, synth, tree


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
                max_size=0.025,
                grace_period=50,
                memory_estimate_period=50,
                splitter=tree.splitter.ExhaustiveSplitter(),
            ),
        ),
        (
            get_classification_data(),
            tree.HoeffdingAdaptiveTreeClassifier(
                leaf_prediction="mc",
                max_size=0.025,
                grace_period=50,
                memory_estimate_period=50,
                splitter=tree.splitter.ExhaustiveSplitter(),
            ),
        ),
        (
            get_classification_data(),
            tree.ExtremelyFastDecisionTreeClassifier(
                leaf_prediction="mc",
                max_size=0.025,
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

    assert model._raw_memory_usage / (2 ** 20) < 0.025


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

    assert model._raw_memory_usage / (2 ** 20) < 0.5


def test_memory_usage_multilabel():
    dataset = datasets.Music().take(500)

    model = tree.LabelCombinationHoeffdingTreeClassifier(
        leaf_prediction="mc",
        splitter=tree.splitter.ExhaustiveSplitter(),
        max_size=1,
        memory_estimate_period=100,
    )
    for x, y in dataset:
        model.learn_one(x, y)

    assert model._raw_memory_usage / (2 ** 20) < 1


def test_memory_usage_multitarget():
    dataset = get_regression_data()

    model = tree.iSOUPTreeRegressor(
        leaf_prediction="mean", max_size=0.5, memory_estimate_period=100,
    )

    for x, y in dataset:
        # Handcrafted targets
        y_ = {0: y, 1: 2 * y, 2: 3 * y}
        model.learn_one(x, y_)

    assert model._raw_memory_usage / (2 ** 20) < 0.5


def test_efdt_split_reevaluation():
    dataset = synth.SEA(seed=7, variant=2).take(500)

    model = tree.ExtremelyFastDecisionTreeClassifier(
        leaf_prediction="nb",
        grace_period=50,
        min_samples_reevaluate=10,
        split_criterion="hellinger",
        split_confidence=0.1,
    )

    max_depth = -1
    for x, y in dataset:
        model.learn_one(x, y)

        if model.height > max_depth:
            max_depth = model.height

    assert model.height != max_depth


def test_drift_adaptation_hatc():
    rng = random.Random(42)
    dataset = iter(synth.Sine(seed=8, classification_function=0, has_noise=True))

    model = tree.HoeffdingAdaptiveTreeClassifier(
        leaf_prediction="mc",
        grace_period=10,
        adwin_confidence=0.1,
        split_confidence=0.1,
        drift_window_threshold=2,
        seed=42,
        max_depth=3,
    )

    for i in range(1000):
        if i % 200 == 0 and i > 0:
            dataset = iter(
                synth.Sine(
                    seed=8, classification_function=rng.randint(0, 3), has_noise=False
                )
            )

        x, y = next(dataset)
        model.learn_one(x, y)

    assert model._n_switch_alternate_trees > 0


def test_drift_adaptation_hatr():
    dataset = synth.Friedman(seed=7).take(500)

    model = tree.HoeffdingAdaptiveTreeRegressor(
        leaf_prediction="model",
        grace_period=50,
        split_confidence=0.1,
        adwin_confidence=0.1,
        drift_window_threshold=10,
        seed=7,
        max_depth=3,
    )

    for i, (x, y) in enumerate(dataset):
        y_ = y
        if i > 250:
            # Emulate an abrupt drift
            y_ = 3 * y
        model.learn_one(x, y_)

    assert model._n_alternate_trees > 0
