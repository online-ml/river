from __future__ import annotations

import pytest

from river import base, datasets, drift, tree


class RecordingDetector(base.DriftDetector):
    def __init__(self):
        super().__init__()
        self.values = []

    def update(self, x):
        self.values.append(x)


def test_error_tracking_detector_input():
    model = tree.LASTClassifier(
        change_detector=RecordingDetector(), leaf_prediction="mc", track_error=True
    )

    model.learn_one({0: 0}, False)
    model.learn_one({0: 1}, True)

    assert model._root.change_detector.values == [False, True]


@pytest.mark.parametrize(
    "split_criterion, expected", [("gini", [0.0, 0.5]), ("info_gain", [0.0, 1.0])]
)
def test_distribution_tracking_detector_input(split_criterion, expected):
    model = tree.LASTClassifier(
        change_detector=RecordingDetector(),
        leaf_prediction="mc",
        split_criterion=split_criterion,
        track_error=False,
    )

    model.learn_one({0: 0}, False)
    model.learn_one({0: 1}, True)

    assert model._root.change_detector.values == expected


@pytest.mark.parametrize("split_criterion", ["gini", "info_gain"])
def test_distribution_tracking_uses_tree_split_criterion(split_criterion):
    model = tree.LASTClassifier(
        change_detector=drift.NoDrift(),
        leaf_prediction="mc",
        split_criterion=split_criterion,
        track_error=False,
    )

    first_leaf = model._new_leaf()
    split_criterion = model._change_detector_split_criterion
    second_leaf = model._new_leaf()

    assert split_criterion is model._change_detector_split_criterion
    assert not hasattr(first_leaf, "split_criterion")
    assert not hasattr(second_leaf, "split_criterion")
    assert model._change_detector_merit({False: 1, True: 1}) == split_criterion.current_merit(
        {False: 1, True: 1}
    )

    first_leaf.learn_one({0: 0}, False, tree=model)
    second_leaf.learn_one({0: 1}, True, tree=model)


@pytest.mark.parametrize("track_error", [True, False])
def test_deterministic_stream_predictions(track_error):
    model = tree.LASTClassifier(
        change_detector=drift.DummyDriftDetector(t_0=10),
        leaf_prediction="mc",
        split_criterion="info_gain",
        track_error=track_error,
    )
    predictions = []

    for x, y in datasets.synth.SEA(seed=42).take(100):
        predictions.append(model.predict_one(x))
        model.learn_one(x, y)

    assert predictions[-20:] == [
        True,
        True,
        False,
        True,
        False,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        False,
        True,
        True,
        True,
        True,
        True,
        True,
        False,
    ]
    assert model.height == 4
    assert model.n_nodes == 11
    assert model.n_leaves == 6
    assert model.predict_proba_one({0: 1, 1: 2, 2: 3}) == {False: 1.0, True: 0.0}


def test_hellinger_distribution_tracking_is_rejected():
    model = tree.LASTClassifier(split_criterion="hellinger", track_error=False)

    with pytest.raises(ValueError, match="cannot estimate the purity"):
        model.learn_one({0: 0}, False)


def test_hellinger_error_tracking_works():
    model = tree.LASTClassifier(
        change_detector=drift.NoDrift(),
        split_criterion="hellinger",
        track_error=True,
    )

    model.learn_one({0: 0}, False)

    assert not hasattr(model._root, "split_criterion")
    assert model._change_detector_split_criterion is None
