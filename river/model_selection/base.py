from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

from river import base, metrics

__all__ = ["ModelSelectionRegressor", "ModelSelectionClassifier"]


class ModelSelector(base.Ensemble, ABC):
    """A generic model selector.

    Parameters
    ----------
    models
    metric

    """

    def __init__(self, models: Iterator[base.Estimator], metric: metrics.base.Metric):
        super().__init__(models)
        for model in models:
            if not metric.works_with(model):
                raise ValueError(
                    f"{metric.__class__.__name__} metric can't be used to evaluate a "
                    + f"{model.__class__.__name__}"
                )
        self.metric = metric

    @property
    @abstractmethod
    def best_model(self):
        """The current best model."""


class ModelSelectionRegressor(ModelSelector, base.Regressor):
    """A model selector for regression.

    Parameters
    ----------
    models
    metric

    """

    def predict_one(self, x):
        if self.best_model is None:
            return self.models[0].predict_one(x)
        return self.best_model.predict_one(x)

    @classmethod
    def _unit_test_params(cls):
        from river import compose, linear_model, optim, preprocessing

        yield {
            "models": [
                compose.Pipeline(
                    preprocessing.StandardScaler(),
                    linear_model.LinearRegression(optimizer=optim.SGD(lr=1e-2)),
                ),
                compose.Pipeline(
                    preprocessing.StandardScaler(),
                    linear_model.LinearRegression(optimizer=optim.SGD(lr=1e-1)),
                ),
            ],
            "metric": metrics.MAE(),
        }
        yield {
            "models": [
                compose.Pipeline(
                    preprocessing.StandardScaler(),
                    linear_model.LinearRegression(optimizer=optim.SGD(lr=lr)),
                )
                for lr in [1e-4, 1e-3, 1e-2, 1e-1]
            ],
            "metric": metrics.MAE(),
        }


class ModelSelectionClassifier(ModelSelector, base.Classifier):
    """A model selector for classification.

    Parameters
    ----------
    models
    metric

    """

    def predict_proba_one(self, x):
        if self.best_model is None:
            return self.models[0].predict_proba_one(x)
        return self.best_model.predict_proba_one(x)

    @classmethod
    def _unit_test_params(cls):
        from river import compose, linear_model, optim, preprocessing

        yield {
            "models": [
                compose.Pipeline(
                    preprocessing.StandardScaler(),
                    linear_model.LogisticRegression(optimizer=optim.SGD(lr=1e-2)),
                ),
                compose.Pipeline(
                    preprocessing.StandardScaler(),
                    linear_model.LogisticRegression(optimizer=optim.SGD(lr=1e-1)),
                ),
            ],
            "metric": metrics.Accuracy(),
        }
        yield {
            "models": [
                compose.Pipeline(
                    preprocessing.StandardScaler(),
                    linear_model.LogisticRegression(optimizer=optim.SGD(lr=lr)),
                )
                for lr in [1e-4, 1e-3, 1e-2, 1e-1]
            ],
            "metric": metrics.Accuracy(),
        }
