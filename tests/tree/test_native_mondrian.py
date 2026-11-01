from __future__ import annotations

import pickle

import pytest

from river import datasets, forest
from river._river_rust.tree import (
    NativeMondrianTreeClassifier,
    NativeMondrianTreeRegressor,
)
from river.tree.mondrian import MondrianTreeClassifier, MondrianTreeRegressor


def random_tree_stream(n_samples: int):
    return list(
        datasets.synth.RandomTree(
            seed_tree=42,
            seed_sample=7,
            n_num_features=8,
            n_cat_features=0,
            max_tree_depth=12,
        ).take(n_samples)
    )


def test_classifier_has_stable_progression() -> None:
    expected = [
        {},
        {1: 1.0},
        {0: 0.2857142857142857, 1: 0.7142857142857143},
        {0: 0.7056451612903226, 1: 0.29435483870967744},
        {0: 0.33249999999999996, 1: 0.6675},
        {0: 0.7976190476190477, 1: 0.20238095238095238},
        {0: 0.7409420289855072, 1: 0.25905797101449274},
        {0: 0.1868906154783668, 1: 0.8131093845216332},
        {0: 0.30268652561247206, 1: 0.6973134743875279},
        {0: 0.2887416637416637, 1: 0.7112583362583363},
    ]
    model = MondrianTreeClassifier(seed=19, max_nodes=511)

    for (x, y), probabilities in zip(random_tree_stream(len(expected)), expected):
        assert model.predict_proba_one(x) == pytest.approx(probabilities)
        model.learn_one(x, y)


def test_regressor_has_stable_progression() -> None:
    expected = [
        0.0,
        1.0,
        0.8112296656009272,
        0.246579412055812,
        0.5618626841877206,
        0.31075273370775225,
        0.1955349096913624,
        0.7334789624024785,
        0.4141776328326348,
        0.6987917253910103,
    ]
    model = MondrianTreeRegressor(seed=19, max_nodes=511)

    for (x, y), prediction in zip(random_tree_stream(len(expected)), expected):
        assert model.predict_one(x) == prediction
        model.learn_one(x, float(y))


@pytest.mark.parametrize(
    "model",
    [MondrianTreeClassifier(seed=42), MondrianTreeRegressor(seed=42)],
)
def test_pickle_continuation(model) -> None:
    stream = random_tree_stream(1_000)
    for x, y in stream[:500]:
        model.learn_one(x, y)

    restored = pickle.loads(pickle.dumps(model))
    for x, y in stream[500:]:
        if isinstance(model, MondrianTreeClassifier):
            assert restored.predict_proba_one(x) == model.predict_proba_one(x)
        else:
            assert restored.predict_one(x) == model.predict_one(x)
        model.learn_one(x, y)
        restored.learn_one(x, y)


@pytest.mark.parametrize(
    "model",
    [MondrianTreeClassifier(seed=42, max_nodes=10), MondrianTreeRegressor(seed=42, max_nodes=10)],
)
def test_dynamic_sparse_features_and_node_limit(model) -> None:
    samples = [
        ({"b": 0.0}, 0),
        ({"a": 1.0, "b": 1.0}, 1),
        ({"c": -1.0}, 2),
        ({"a": 2.0, "c": 1.0}, 0),
    ]
    for _ in range(100):
        for x, y in samples:
            model.learn_one(x, y)

    assert model._n_nodes <= 10
    assert model.predict_one({"a": 0.5}) is not None


def test_regressor_does_not_retain_input_mapping() -> None:
    model = MondrianTreeRegressor(seed=42)
    x = {"a": 1.0, "b": 2.0}
    model.learn_one(x, 3.0)
    expected = model.predict_one({"a": 1.0, "b": 2.0})

    x["a"] = 100.0

    assert model.predict_one({"a": 1.0, "b": 2.0}) == expected


def test_public_models_use_native_state() -> None:
    classifier = MondrianTreeClassifier(seed=42)
    regressor = MondrianTreeRegressor(seed=42)
    x, y = random_tree_stream(1)[0]

    classifier.learn_one(x, y)
    regressor.learn_one(x, float(y))

    assert isinstance(classifier._native_model, NativeMondrianTreeClassifier)
    assert isinstance(regressor._native_model, NativeMondrianTreeRegressor)


def test_classifier_preserves_observed_class_order_for_ties() -> None:
    model = MondrianTreeClassifier(use_aggregation=False, max_nodes=1, seed=42)
    model.learn_one({"x": 0.0}, "z")
    model.learn_one({"x": 0.0}, "a")

    probabilities = model.predict_proba_one({"x": 0.0})

    assert list(probabilities) == ["z", "a"]
    assert model.predict_one({"x": 0.0}) == "z"


def test_amf_uses_native_members() -> None:
    model = forest.AMFClassifier(n_estimators=3, seed=42)
    for x, y in random_tree_stream(100):
        model.learn_one(x, y)

    assert all(isinstance(tree._native_model, NativeMondrianTreeClassifier) for tree in model)
