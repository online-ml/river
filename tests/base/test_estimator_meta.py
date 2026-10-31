from __future__ import annotations

import pytest

from river import base, linear_model, preprocessing
from river.base import AnomalyDetector


def test_direct_model_categories() -> None:
    classifier = linear_model.LogisticRegression()
    regressor = linear_model.LinearRegression()
    transformer = preprocessing.StandardScaler()

    assert isinstance(classifier, base.Classifier)
    assert not isinstance(classifier, base.Regressor)
    assert isinstance(regressor, base.Regressor)
    assert not isinstance(regressor, base.Classifier)
    assert isinstance(transformer, base.Transformer)
    assert not isinstance("not a model", base.Classifier)
    assert not isinstance(42, base.Regressor)
    assert not isinstance(None, base.Estimator)


@pytest.mark.parametrize(
    ("terminal", "category"),
    [
        (linear_model.LogisticRegression(), base.Classifier),
        (linear_model.LinearRegression(), base.Regressor),
        (preprocessing.MinMaxScaler(), base.Transformer),
    ],
)
def test_pipeline_category_is_explicit(terminal, category) -> None:
    model = preprocessing.StandardScaler() | terminal

    assert isinstance(model, base.Estimator)
    assert not isinstance(model, category)
    assert isinstance(base._unwrap(model), category)


def test_composing_pipelines_flattens_them() -> None:
    inner = preprocessing.StandardScaler() | linear_model.LogisticRegression()
    outer = preprocessing.MinMaxScaler() | inner

    assert len(outer) == 3
    assert isinstance(base._unwrap(outer), base.Classifier)


def test_anomaly_detector_category_is_explicit() -> None:
    from river import anomaly

    detector = anomaly.HalfSpaceTrees()
    model = preprocessing.StandardScaler() | detector

    assert isinstance(detector, AnomalyDetector)
    assert not isinstance(model, AnomalyDetector)
    assert isinstance(base._unwrap(model), AnomalyDetector)


def test_nominal_subclasses() -> None:
    assert issubclass(base.Classifier, base.Estimator)
    assert not issubclass(base.Regressor, base.Classifier)


def test_abstract_method_enforcement() -> None:
    with pytest.raises(TypeError):
        base.Classifier()  # type: ignore[abstract]

    with pytest.raises(TypeError):
        base.Regressor()  # type: ignore[abstract]
