from __future__ import annotations

import math

from river import base, linear_model, metrics, neighbors


class HoeffdingRaceClassifier(base.Classifier):
    """
    HoeffdingRace-based model selection for Classification.

    Each models is associated to a performance (here its accuracy). When the model is considered too inaccurate by the hoeffding bound,
    the model is removed.

    Parameters
    ----------
    models
        The models to select from.
    metric
        The metric that is used to measure the performance of each model.
    delta
        Hoeffding bound precision.


    Tests on Binary Classification

    >>> from river import model_selection
    >>> from river import linear_model, neighbors, tree, metrics, datasets

    >>> hoeffding_race = model_selection.HoeffdingRaceClassifier(
    ...     models = {
    ...     "KNN": neighbors.KNNClassifier(),
    ...     "Log_Reg":linear_model.LogisticRegression()},
    ...     metric=metrics.Accuracy(),
    ...     delta=0.05
    ... )
    >>> dataset = datasets.Phishing()
    >>> for x, y in dataset:
    ...     hoeffding_race.learn_one(x, y)
    ...     if hoeffding_race.single_model_remaining():
    ...             break
    ...
    >>> hoeffding_race.remaining_models
    ['KNN']
    """

    def __init__(
        self,
        models={"KNN": neighbors.KNNClassifier(), "Log_Reg": linear_model.LogisticRegression()},
        delta=0.05,
        metric=metrics.Accuracy(),
    ):
        self.models = models
        self.delta = delta
        self.metric = metric
        self.n = 0
        self.model_metrics = {name: metric.clone() for name in models.keys()}
        self.model_performance = {name: 0 for name in models.keys()}
        self.remaining_models = [i for i in models.keys()]

    def hoeffding_bound(self, n):
        """
        Computes the hoeffding bound according to n, the number of iterations done.

        """
        return math.sqrt((math.log(1 / self.delta)) / (2 * n))

    def learn_one(self, x, y):
        best_perf = max(self.model_performance.values()) if self.n > 0 else 0
        self.n = self.n + 1

        for name in list(self.remaining_models):
            y_pred = self.models[name].predict_one(x)
            self.models[name].learn_one(x, y)

            # Update performance

            self.model_metrics[name].update(y, y_pred)
            self.model_performance[name] = self.model_metrics[name].get()

            if self.model_performance[name] + self.hoeffding_bound(self.n) < best_perf:
                self.remaining_models.remove(name)
                if len(self.remaining_models) == 1:
                    break

    def predict_one(self, x):
        if len(self.remaining_models) == 1:
            return self.models[list(self.remaining_models)[0]].predict_one(x)
        return None

    def single_model_remaining(self):
        return len(self.remaining_models) == 1


class HoeffdingRaceRegressor(base.Regressor):
    """
    HoeffdingRace-based model selection for regression.

    Each models is associated to a performance (here its accuracy). When the model is considered too inaccurate by the hoeffding bound,
    the model is removed.

    Parameters
    ----------
    models
        The models to select from.
    metric
        The metric that is used to measure the performance of each model.
    delta
        Hoeffding bound precision.

    Tests on Regression models
    >>> from river import linear_model, neighbors, tree, metrics, datasets, model_selection
    >>> hoeffding_race = model_selection.HoeffdingRaceRegressor(
    ... models = {"KNN": neighbors.KNNRegressor(),
    ...           "Log_Reg":linear_model.LinearRegression()},
    ...           metric=metrics.MAE(),
    ...           delta=0.05)
    >>> dataset = datasets.ChickWeights()
    >>> for x, y in dataset:
    ...     hoeffding_race.learn_one(x, y)
    ...     if hoeffding_race.single_model_remaining():
    ...         break
    ...
    >>> print(hoeffding_race.remaining_models)
    ['Log_Reg']

    """

    def __init__(
        self,
        models={"KNN": neighbors.KNNRegressor(), "Log_Reg": linear_model.LinearRegression()},
        delta=0.05,
        metric=metrics.MAE(),
    ):
        self.models = models
        self.delta = delta
        self.metric = metric
        self.n = 0
        self.model_metrics = {name: metric.clone() for name in models.keys()}
        self.model_performance = {name: 0 for name in models.keys()}
        self.remaining_models = [i for i in models.keys()]

    def hoeffding_bound(self, n):
        return math.sqrt((math.log(1 / self.delta)) / (2 * n))

    def learn_one(self, x, y):
        best_perf = max(self.model_performance.values()) if self.n > 0 else 0
        self.n = self.n + 1

        for name in list(self.remaining_models):
            y_pred = self.models[name].predict_one(x)
            self.models[name].learn_one(x, y)

            # Update performance

            self.model_metrics[name].update(y, y_pred)
            self.model_performance[name] = self.model_metrics[name].get()

            if self.model_performance[name] + self.hoeffding_bound(self.n) < best_perf:
                self.remaining_models.remove(name)
                if len(self.remaining_models) == 1:
                    break

    def predict_one(self, x):
        if len(self.remaining_models) == 1:
            return self.models[list(self.remaining_models)[0]].predict_one(x)
        return None

    def single_model_remaining(self):
        """
        Method to be able to know if the "race" has ended.
        """
        return len(self.remaining_models) == 1
