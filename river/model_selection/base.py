from abc import abstractproperty
from typing import Iterator

from river import compose, linear_model, metrics, optim, preprocessing
from river.base import Classifier, Ensemble, Estimator, Regressor
from river.metrics import Metric


class ModelSelector(Ensemble):
    def __init__(self, models: Iterator[Estimator], metric: Metric):
        super().__init__(models)
        for model in models:
            if not metric.works_with(model):
                raise ValueError(
                    f"{metric.__class__.__name__} metric can't be used to evaluate a "
                    + f"{model.__class__.__name__}"
                )
        self.metric = metric

    @abstractproperty
    def best_model(self):
        """The current best model."""


class ModelSelectionRegressor(ModelSelector, Regressor):
    def predict_one(self, x):
        return self.best_model.predict_one(x)

    @classmethod
    def _unit_test_params(cls):
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


class ModelSelectionClassifier(ModelSelector, Classifier):
    def predict_proba_one(self, x):
        return self.best_model.predict_proba_one(x)
