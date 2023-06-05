from __future__ import annotations

import pytest

from river import datasets, tree
from river.datasets import synth


def get_regression_data():
    return iter(synth.Friedman(seed=42).take(200))


@pytest.mark.parametrize(
    "dataset, splitter",
    [
        (datasets.Phishing(), tree.splitter.ExhaustiveSplitter()),
        (datasets.Phishing(), tree.splitter.HistogramSplitter()),
        (datasets.Phishing(), tree.splitter.GaussianSplitter()),
    ],
)
def test_class_splitter(dataset, splitter):
    model = tree.HoeffdingTreeClassifier(
        splitter=splitter, grace_period=10, leaf_prediction="mc", delta=0.1
    )

    for x, y in dataset:
        model.learn_one(x, y)

    assert model.height > 0


@pytest.mark.parametrize(
    "dataset, splitter",
    [
        (get_regression_data(), tree.splitter.EBSTSplitter()),
        (get_regression_data(), tree.splitter.TEBSTSplitter()),
        (get_regression_data(), tree.splitter.QOSplitter()),
        (get_regression_data(), tree.splitter.QOSplitter(allow_multiway_splits=True)),
    ],
)
def test_reg_splitter(dataset, splitter):
    model = tree.HoeffdingTreeRegressor(
        splitter=splitter, grace_period=20, delta=0.1, leaf_prediction="mean"
    )

    for x, y in dataset:
        model.learn_one(x, y)

    assert model.height > 0


def test_nominal_reg_splitter():
    dataset = synth.Mv(seed=42).take(200)
    model = tree.HoeffdingTreeRegressor(grace_period=10, leaf_prediction="mean")

    for x, y in dataset:
        model.learn_one(x, y)

    assert model.height > 0

    # Evaluates nominal binary splits
    dataset = synth.Mv(seed=42).take(200)
    model = tree.HoeffdingTreeRegressor(grace_period=10, leaf_prediction="mean", binary_split=True)

    for x, y in dataset:
        model.learn_one(x, y)

    assert model.height > 0
