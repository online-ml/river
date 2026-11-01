from __future__ import annotations

import pickle

import numpy as np
import pytest

from river import datasets, ensemble, forest, tree
from river._river_rust.tree import NativeHoeffdingTreeClassifier


class PythonGaussianSplitter(tree.splitter.GaussianSplitter):
    pass


def make_stream(n_samples: int, n_features: int = 6) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    x = rng.normal(size=(n_samples, n_features))
    y = (x[:, 0] + 0.75 * x[:, 1] - 0.5 * x[:, 2] > 0).astype(np.int64)
    return x, y


def progressive_predictions(model, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    predictions = []
    for row, label in zip(x, y):
        values = row.tolist()
        predictions.append(model.predict_one(values))
        model.learn_one(values, int(label))
    return np.asarray(predictions)


def test_native_hoeffding_tree_is_deterministic() -> None:
    x, y = make_stream(2_000)
    first = NativeHoeffdingTreeClassifier(6, grace_period=50, seed=42)
    second = NativeHoeffdingTreeClassifier(6, grace_period=50, seed=42)

    first_predictions = progressive_predictions(first, x, y)
    second_predictions = progressive_predictions(second, x, y)

    assert np.array_equal(first_predictions, second_predictions)
    assert first.n_nodes() == second.n_nodes()


def test_native_hoeffding_tree_accuracy_tracks_python_tree() -> None:
    x, y = make_stream(5_000)
    native = NativeHoeffdingTreeClassifier(6, grace_period=50, seed=42)
    native_predictions = progressive_predictions(native, x, y)

    python = tree.HoeffdingTreeClassifier(grace_period=50, splitter=PythonGaussianSplitter())
    python_predictions = []
    for row, label in zip(x, y):
        sample = dict(enumerate(row))
        python_predictions.append(int(bool(python.predict_one(sample))))
        python.learn_one(sample, bool(label))
    python_predictions_array = np.asarray(python_predictions, dtype=np.int64)

    native_accuracy = np.mean(native_predictions == y)
    python_accuracy = np.mean(python_predictions_array == y)
    assert native_accuracy > 0.85
    assert abs(native_accuracy - python_accuracy) < 0.05


@pytest.mark.parametrize(
    ("leaf_prediction", "nb_threshold"),
    [("mc", 0), ("nb", 20), ("nba", 0)],
)
def test_native_hoeffding_tree_leaf_probabilities_match_python(
    leaf_prediction: str, nb_threshold: int
) -> None:
    x, y = make_stream(300)
    x[:2] = [[-1.0] * 6, [1.0] * 6]
    y[:2] = [0, 1]
    native = NativeHoeffdingTreeClassifier(
        6,
        grace_period=1e9,
        leaf_prediction=leaf_prediction,
        nb_threshold=nb_threshold,
        seed=42,
    )
    python = tree.HoeffdingTreeClassifier(
        grace_period=int(1e9),
        leaf_prediction=leaf_prediction,
        nb_threshold=nb_threshold,
        splitter=PythonGaussianSplitter(),
    )

    for row, label in zip(x, y):
        sample = dict(enumerate(row))
        python_proba = python.predict_proba_one(sample)
        native_proba = native.predict_proba_one(row.tolist())
        assert native_proba == pytest.approx(
            [python_proba.get(False, 0.0), python_proba.get(True, 0.0)]
        )
        native.learn_one(row.tolist(), int(label))
        python.learn_one(sample, bool(label))


@pytest.mark.parametrize("split_criterion", ["info_gain", "gini", "hellinger"])
def test_native_hoeffding_tree_split_criteria(split_criterion: str) -> None:
    x, y = make_stream(5_000)
    model = NativeHoeffdingTreeClassifier(
        6,
        grace_period=50,
        split_criterion=split_criterion,
        seed=42,
    )

    predictions = progressive_predictions(model, x, y)

    assert np.mean(predictions == y) > 0.8
    assert model.n_nodes() > 1
    assert model.n_nodes() == model.n_branches() + model.n_leaves()
    assert model.n_leaves() == model.n_branches() + 1
    assert model.n_active_leaves() <= model.n_leaves()
    assert model.height() > 0
    assert model.total_weight_seen() == len(y)


def test_native_hoeffding_tree_multiclass() -> None:
    rng = np.random.default_rng(42)
    x = rng.normal(size=(5_000, 5))
    y = np.argmax(x[:, :3], axis=1).astype(np.int64)
    model = NativeHoeffdingTreeClassifier(5, n_classes=3, grace_period=50, seed=42)

    predictions = progressive_predictions(model, x, y)
    probabilities = model.predict_proba_one(x[-1].tolist())

    assert np.mean(predictions == y) > 0.8
    assert len(probabilities) == 3
    assert sum(probabilities) == pytest.approx(1.0)


def test_native_hoeffding_tree_weighted_learning() -> None:
    weighted = NativeHoeffdingTreeClassifier(2, grace_period=1e9, leaf_prediction="nb", seed=42)
    repeated = NativeHoeffdingTreeClassifier(2, grace_period=1e9, leaf_prediction="nb", seed=42)
    samples = [([-1.0, 0.5], 0, 3), ([2.0, -1.0], 1, 2), ([0.5, 1.5], 1, 4)]

    for x, y, weight in samples:
        weighted.learn_one(x, y, w=weight)
        for _ in range(weight):
            repeated.learn_one(x, y)

    for x, _, _ in samples:
        assert weighted.predict_proba_one(x) == pytest.approx(repeated.predict_proba_one(x))
    assert weighted.total_weight_seen() == repeated.total_weight_seen()


def test_native_hoeffding_tree_pickle_round_trip() -> None:
    x, y = make_stream(2_000)
    model = NativeHoeffdingTreeClassifier(6, grace_period=50, seed=42)
    progressive_predictions(model, x[:1_000], y[:1_000])

    restored = pickle.loads(pickle.dumps(model))
    expected = progressive_predictions(model, x[1_000:], y[1_000:])
    actual = progressive_predictions(restored, x[1_000:], y[1_000:])

    assert np.array_equal(actual, expected)
    assert restored.n_nodes() == model.n_nodes()


def test_public_hoeffding_tree_memory_includes_native_allocations() -> None:
    model = tree.HoeffdingTreeClassifier(grace_period=50)
    x, y = make_stream(2_000)

    for row, label in zip(x, y):
        model.learn_one(dict(enumerate(row)), bool(label))

    assert model._raw_memory_usage >= len(pickle.dumps(model._native_model))


@pytest.mark.parametrize(
    ("args", "kwargs"),
    [
        ((0,), {}),
        ((3,), {"n_classes": 1}),
        ((3,), {"max_features": 4}),
        ((3,), {"n_splits": 0}),
        ((3,), {"split_criterion": "unknown"}),
        ((3,), {"leaf_prediction": "unknown"}),
        ((3,), {"n_classes": 3, "split_criterion": "hellinger"}),
        ((3,), {"tau": np.nan}),
        ((3,), {"nb_threshold": np.inf}),
    ],
)
def test_native_tree_rejects_invalid_parameters(args, kwargs) -> None:
    with pytest.raises(ValueError):
        NativeHoeffdingTreeClassifier(*args, **kwargs)


def test_native_tree_rejects_invalid_data() -> None:
    model = NativeHoeffdingTreeClassifier(3)
    with pytest.raises(ValueError):
        model.learn_one([1.0, 2.0], 0)
    with pytest.raises((TypeError, ValueError, OverflowError)):
        model.learn_one([1.0, 2.0, 3.0], -1)
    with pytest.raises(ValueError):
        model.learn_one([1.0, 2.0, 3.0], 2)
    model.learn_one([1.0, np.nan, 3.0], 1)
    with pytest.raises(ValueError):
        model.learn_one([1.0, 2.0, 3.0], 1, w=np.inf)


def test_public_hoeffding_tree_uses_native_backend() -> None:
    model = tree.HoeffdingTreeClassifier(grace_period=2)

    model.learn_one({"a": 0.0, "b": 1.0}, "first")
    model.learn_one({"a": 1.0}, "second")
    model.learn_one({"a": 2.0, "c": -1.0}, "third")

    assert isinstance(model._native_model, NativeHoeffdingTreeClassifier)
    assert set(model.predict_proba_one({"c": 1.0})) == {"first", "second", "third"}
    assert model.n_nodes == model.n_branches + model.n_leaves
    assert model.summary["total_observed_weight"] == 3.0

    restored = pickle.loads(pickle.dumps(model))
    assert restored.predict_proba_one({"a": 0.5}) == model.predict_proba_one({"a": 0.5})


def test_public_hoeffding_tree_inspection_uses_native_snapshot() -> None:
    x, y = make_stream(500, n_features=3)
    model = tree.HoeffdingTreeClassifier(grace_period=5, split_criterion="gini", delta=0.01)

    for row, label in zip(x, y):
        model.learn_one(dict(enumerate(row)), bool(label))

    sample = dict(enumerate(x[-1]))
    assert model.n_nodes > 1
    assert model.debug_one(sample)
    assert model.draw().body
    pytest.importorskip("pandas")
    assert len(model.to_dataframe()) == model.n_nodes


@pytest.mark.parametrize(
    "kwargs",
    [
        {"splitter": tree.splitter.HistogramSplitter()},
        {"nominal_attributes": ["a"]},
        {"split_criterion": "hellinger"},
        {"max_size": 1.0},
    ],
)
def test_public_hoeffding_tree_keeps_python_backend_for_extended_options(kwargs) -> None:
    model = tree.HoeffdingTreeClassifier(**kwargs)
    assert not model._native_enabled


def test_public_adaptive_random_forest_composes_native_trees() -> None:
    model = forest.ARFClassifier(n_models=3, seed=42)

    for x, y in datasets.synth.Hyperplane(seed=42).take(200):
        model.learn_one(x, y)

    assert len(model) == 3
    assert model.models is model.data
    assert all(isinstance(member._native_model, NativeHoeffdingTreeClassifier) for member in model)
    assert model.n_warnings_detected() == sum(model.n_warnings_detected(i) for i in range(3))
    assert model.n_drifts_detected() == sum(model.n_drifts_detected(i) for i in range(3))

    restored = pickle.loads(pickle.dumps(model))
    assert len(restored) == 3
    assert restored.predict_proba_one({"x0": 0.5}) == model.predict_proba_one({"x0": 0.5})


def test_public_adaptive_random_forest_matches_python_trees() -> None:
    stream = list(datasets.synth.ConceptDriftStream(seed=42, position=500, width=40).take(1_000))
    native = forest.ARFClassifier(seed=8, leaf_prediction="mc")
    python = forest.ARFClassifier(
        seed=8,
        leaf_prediction="mc",
        splitter=PythonGaussianSplitter(),
    )

    for x, y in stream:
        assert native.predict_one(x) == python.predict_one(x)
        native.learn_one(x, y)
        python.learn_one(x, y)

    assert native.n_warnings_detected() == python.n_warnings_detected()
    assert native.n_drifts_detected() == python.n_drifts_detected()


@pytest.mark.parametrize(
    "model",
    [
        ensemble.BaggingClassifier(tree.HoeffdingTreeClassifier(), n_models=3, seed=42),
        ensemble.ADWINBaggingClassifier(tree.HoeffdingTreeClassifier(), n_models=3, seed=42),
        ensemble.AdaBoostClassifier(tree.HoeffdingTreeClassifier(), n_models=3, seed=42),
        ensemble.SRPClassifier(n_models=3, seed=42),
    ],
)
def test_hoeffding_tree_ensembles_use_native_members(model) -> None:
    for x, y in datasets.synth.Hyperplane(seed=42).take(100):
        model.learn_one(x, y)

    members = [member.model if hasattr(member, "model") else member for member in model]
    assert all(
        isinstance(member._native_model, NativeHoeffdingTreeClassifier) for member in members
    )
