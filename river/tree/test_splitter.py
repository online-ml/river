import pytest

from river import datasets
from river import synth
from river import tree


def get_regression_data():
    return iter(synth.Friedman(seed=42).take(200))


@pytest.mark.parametrize(
    'dataset, splitter',
    [
        (datasets.Phishing(), tree.splitter.ExhaustiveSplitter()),
        (datasets.Phishing(), tree.splitter.HistogramSplitter()),
        (datasets.Phishing(), tree.splitter.GaussianSplitter())
    ]
)
def test_class_splitter(dataset, splitter):
    model = tree.HoeffdingTreeClassifier(
        splitter=splitter,
        grace_period=10,
        leaf_prediction="mc",
        split_confidence=0.1
    )

    for x, y in dataset:
        model.learn_one(x, y)

    assert model.depth > 0


@pytest.mark.parametrize(
    'dataset, splitter',
    [
        (get_regression_data(), tree.splitter.EBSTSplitter()),
        (get_regression_data(), tree.splitter.TEBSTSplitter()),
        (get_regression_data(), tree.splitter.QOSplitter())
    ]
)
def test_reg_splitter(dataset, splitter):
    model = tree.HoeffdingTreeRegressor(
        splitter=splitter,
        grace_period=20,
        split_confidence=0.1,
        leaf_prediction="mean"
    )

    for x, y in dataset:
        model.learn_one(x, y)

    assert model.depth > 0
