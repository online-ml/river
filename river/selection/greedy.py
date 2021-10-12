from copy import deepcopy
from typing import List

from river.base import Regressor
from river.metrics import MAE, RegressionMetric

from .base import ModelSelector


class GreedySelectionRegressor(ModelSelector, Regressor):
    """Greedy selection regressor.

    This selection method simply updates each model at each time step.

    Parameters
    ----------
    models
        The models to select from.
    metric
        The metric that is used to measure the performance of each model.

    Examples
    --------

    >>> from river import compose
    >>> from river import datasets
    >>> from river import evaluate
    >>> from river import linear_model
    >>> from river import metrics
    >>> from river import optim
    >>> from river import preprocessing
    >>> from river import selection

    >>> models = [
    ...     compose.Pipeline(
    ...         preprocessing.StandardScaler(),
    ...         linear_model.LinearRegression(optimizer=optim.SGD(lr=lr))
    ...     )
    ...     for lr in [1e-4, 1e-3, 1e-2, 1e-1]
    ... ]

    >>> dataset = datasets.TrumpApproval()
    >>> model = selection.GreedySelectionRegressor(models, metric)
    >>> metric = metrics.MAE()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    MAE: 1.416792

    """

    def __init__(self, models: List[Regressor], metric: RegressionMetric = None):
        if metric is None:
            metric = MAE()
        super().__init__(models, metric)
        self.metrics = [deepcopy(metric) for _ in range(len(self))]

    def learn_one(self, x, y):
        for model, metric in zip(self, self.metrics):
            y_pred = model.predict_one(x)
            metric.update(y, y_pred)
            model.learn_one(x, y)
        return self

    @property
    def best_model(self):
        best_model_idx = min(range(len(self)), key=lambda i: self.metrics[i].get())
        return self.models[best_model_idx]
