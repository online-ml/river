from __future__ import annotations

import pytest

from river import drift, tree


@pytest.mark.parametrize("split_criterion, expected_merit", [("gini", 0.5), ("info_gain", 1.0)])
def test_distribution_tracking_uses_shared_split_criterion(split_criterion, expected_merit):
    model = tree.LASTClassifier(
        change_detector=drift.NoDrift(),
        leaf_prediction="mc",
        split_criterion=split_criterion,
        track_error=False,
    )

    first_leaf = model._new_leaf()
    criterion = model._change_detector_split_criterion
    second_leaf = model._new_leaf()

    assert criterion is model._change_detector_split_criterion
    assert not hasattr(first_leaf, "split_criterion")
    assert not hasattr(second_leaf, "split_criterion")
    assert model._change_detector_merit({False: 1, True: 1}) == expected_merit


def test_error_tracking_does_not_initialize_split_criterion():
    model = tree.LASTClassifier(
        change_detector=drift.NoDrift(), leaf_prediction="mc", track_error=True
    )

    leaf = model._new_leaf()
    leaf.learn_one({0: 0}, False, tree=model)

    assert model._change_detector_split_criterion is None
    assert not hasattr(leaf, "split_criterion")
