from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np
import pytest
from sklearn import datasets
from sklearn.utils import Bunch

from river import stream


@pytest.mark.parametrize(
    "loader",
    [
        datasets.load_iris,
        datasets.load_wine,
        datasets.load_diabetes,
        datasets.load_breast_cancer,
    ],
)
@pytest.mark.parametrize("as_frame", [False, True])
def test_iter_sklearn_dataset(loader: Callable[..., Bunch], as_frame: bool) -> None:
    if as_frame:
        pytest.importorskip("pandas")
    dataset = loader(as_frame=as_frame)
    reference = loader(as_frame=False)
    expected = [
        (dict(zip(reference.feature_names, x.tolist())), y.item())
        for x, y in zip(reference.data, reference.target)
    ]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        actual = list(stream.iter_sklearn_dataset(dataset))
        shuffled = list(stream.iter_sklearn_dataset(dataset, shuffle=True, seed=17))
        repeated = list(stream.iter_sklearn_dataset(dataset, shuffle=True, seed=17))

    assert actual == expected
    assert shuffled == repeated
    assert shuffled != actual
    assert sorted(shuffled, key=lambda row: (tuple(row[0].values()), row[1])) == sorted(
        expected, key=lambda row: (tuple(row[0].values()), row[1])
    )
    assert not any(issubclass(w.category, DeprecationWarning) for w in caught)


def test_iter_sklearn_dataset_without_feature_names() -> None:
    dataset = Bunch(data=np.array([[1, 2], [3, 4]]), target=np.array([0, 1]))

    assert list(stream.iter_sklearn_dataset(dataset)) == [
        ({0: 1, 1: 2}, 0),
        ({0: 3, 1: 4}, 1),
    ]


def test_iter_sklearn_dataset_frame_columns() -> None:
    pd = pytest.importorskip("pandas")
    dataset = Bunch(
        data=pd.DataFrame({"left": [1, 2], "right": [3, 4]}, index=[8, 5]),
        target=pd.Series([0, 1], index=[8, 5]),
        feature_names=["ignored_left", "ignored_right"],
    )

    assert list(stream.iter_sklearn_dataset(dataset)) == [
        ({"left": 1, "right": 3}, 0),
        ({"left": 2, "right": 4}, 1),
    ]


def test_iter_sklearn_dataset_multioutput_frame() -> None:
    pd = pytest.importorskip("pandas")
    dataset = Bunch(
        data=pd.DataFrame({"feature": [1, 2]}),
        target=pd.DataFrame({"first": [0, 1], "second": [2, 3]}),
        feature_names=["feature"],
    )

    assert list(stream.iter_sklearn_dataset(dataset)) == [
        ({"feature": 1}, {"first": 0, "second": 2}),
        ({"feature": 2}, {"first": 1, "second": 3}),
    ]
