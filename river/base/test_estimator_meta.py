"""Tests for EstimatorMeta metaclass that makes isinstance() work with pipelines."""

from __future__ import annotations

import pytest

from river import base, linear_model, preprocessing
from river.anomaly.base import AnomalyDetector


class TestIsinstanceWithDirectModels:
    """isinstance should work normally for non-pipeline models."""

    def test_classifier(self) -> None:
        model = linear_model.LogisticRegression()
        assert isinstance(model, base.Classifier)
        assert not isinstance(model, base.Regressor)

    def test_regressor(self) -> None:
        model = linear_model.LinearRegression()
        assert isinstance(model, base.Regressor)
        assert not isinstance(model, base.Classifier)

    def test_transformer(self) -> None:
        model = preprocessing.StandardScaler()
        assert isinstance(model, base.Transformer)

    def test_non_estimator(self) -> None:
        assert not isinstance("not a model", base.Classifier)
        assert not isinstance(42, base.Regressor)
        assert not isinstance(None, base.Estimator)


class TestIsinstanceWithPipelines:
    """isinstance should transparently unwrap pipelines."""

    def test_pipeline_with_classifier(self) -> None:
        model = preprocessing.StandardScaler() | linear_model.LogisticRegression()
        assert isinstance(model, base.Classifier)
        assert not isinstance(model, base.Regressor)

    def test_pipeline_with_regressor(self) -> None:
        model = preprocessing.StandardScaler() | linear_model.LinearRegression()
        assert isinstance(model, base.Regressor)
        assert not isinstance(model, base.Classifier)

    def test_pipeline_is_still_estimator(self) -> None:
        model = preprocessing.StandardScaler() | linear_model.LogisticRegression()
        assert isinstance(model, base.Estimator)

    def test_pipeline_with_transformer_only(self) -> None:
        model = preprocessing.StandardScaler() | preprocessing.MinMaxScaler()
        assert isinstance(model, base.Transformer)
        assert not isinstance(model, base.Classifier)

    def test_nested_pipeline(self) -> None:
        """Nested pipelines should be unwrapped recursively."""
        inner = preprocessing.StandardScaler() | linear_model.LogisticRegression()
        outer = preprocessing.MinMaxScaler() | inner
        assert isinstance(outer, base.Classifier)
        assert not isinstance(outer, base.Regressor)


class TestIsinstanceWithAnomalyDetectors:
    """isinstance should work for anomaly detection classes via pipelines."""

    def test_anomaly_detector_direct(self) -> None:
        from river import anomaly

        model = anomaly.HalfSpaceTrees()
        assert isinstance(model, AnomalyDetector)

    def test_anomaly_detector_pipeline(self) -> None:
        from river import anomaly

        model = preprocessing.StandardScaler() | anomaly.HalfSpaceTrees()
        assert isinstance(model, AnomalyDetector)
        assert not isinstance(model, base.Classifier)


class TestIssubclass:
    """issubclass should still work correctly."""

    def test_classifier_is_estimator(self) -> None:
        assert issubclass(base.Classifier, base.Estimator)

    def test_regressor_is_not_classifier(self) -> None:
        assert not issubclass(base.Regressor, base.Classifier)


class TestAbstractMethodEnforcement:
    """The metaclass should not break ABC abstract method enforcement."""

    def test_cannot_instantiate_abstract(self) -> None:
        with pytest.raises(TypeError):
            base.Classifier()  # type: ignore[abstract]

        with pytest.raises(TypeError):
            base.Regressor()  # type: ignore[abstract]
