from __future__ import annotations

import pytest

from river import anomaly, base, linear_model, preprocessing


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
        (anomaly.HalfSpaceTrees(), base.AnomalyDetector),
    ],
)
def test_pipeline_category_is_explicit(terminal, category) -> None:
    model = preprocessing.StandardScaler() | terminal

    assert isinstance(model, base.Estimator)
    assert not isinstance(model, category)
    assert isinstance(base.unwrap_estimator(model), category)


def test_nested_pipeline_unwrap() -> None:
    inner = preprocessing.StandardScaler() | linear_model.LogisticRegression()
    outer = preprocessing.MinMaxScaler() | inner

    assert isinstance(base.unwrap_estimator(outer), base.Classifier)


def test_nominal_subclasses() -> None:
    assert issubclass(base.Classifier, base.Estimator)
    assert not issubclass(base.Regressor, base.Classifier)


def test_abstract_method_enforcement() -> None:
    with pytest.raises(TypeError):
        base.Classifier()  # type: ignore[abstract]

    with pytest.raises(TypeError):
        base.Regressor()  # type: ignore[abstract]
