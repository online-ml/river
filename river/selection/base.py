from abc import abstractproperty
from typing import Iterator

from river.base import Ensemble, Estimator
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
