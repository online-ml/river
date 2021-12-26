from abc import ABC, abstractmethod
from typing import Iterator, List

from river import compose, linear_model, metrics, optim, preprocessing
from river._bandit import Bandit, BanditPolicy
from river.base import Classifier, Ensemble, Estimator, Regressor
from river.metrics import Metric, RegressionMetric


class ModelSelector(Ensemble, ABC):
    """

    Parameters
    ----------
    models
    metric

    """

    def __init__(self, models: Iterator[Estimator], metric: Metric):
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


class BanditRegressor(ModelSelectionRegressor):
    def __init__(
        self,
        models: List[Regressor],
        metric: RegressionMetric,
        policy: BanditPolicy,
    ):
        super().__init__(models, metric)
        self.bandit = Bandit(n_arms=len(models), metric=metric)
        self.policy = policy

    @property
    def best_model(self):
        return self[self.bandit.best_arm.index]

    def learn_one(self, x, y):
        for arm in self.policy.pull(self.bandit):
            model = self[arm.index]
            y_pred = model.predict_one(x)
            self.bandit.update(arm, y_true=y, y_pred=y_pred)
            model.learn_one(x, y)
        return self
