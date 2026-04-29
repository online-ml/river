from __future__ import annotations

import abc
import math

from river import metrics
from river.model_selection.base import ModelSelectionClassifier, ModelSelectionRegressor


class HoeffdingRace(abc.ABC):
    r"""Hoeffding Race model selection.

    Uses the Hoeffding bound to progressively eliminate underperforming models with
    statistical confidence. Each model is evaluated on every observation. After each
    observation, models whose lower bound on error exceeds the upper bound of the best
    model are eliminated from the race.

    The Hoeffding bound guarantees that the estimated mean of `n` bounded observations
    is within epsilon of the true mean with confidence `1 - delta`:

    $$\epsilon = \sqrt{\frac{B^2 (\log(2nm) - \log(\Delta))}{n}}$$

    where `B` is the range of the loss, `n` is the number of observations, `m` is the
    number of models, and `Delta` is the overall confidence parameter.

    Parameters
    ----------
    models
        The models to race against each other.
    metric
        Metric used for comparing models.
    delta
        Confidence parameter. Smaller values make elimination harder (more conservative).
        The algorithm is correct with probability at least `1 - delta`.
    loss_range
        The range of the loss values (max - min). For metrics bounded in [0, 1] like
        accuracy or zero-one loss, this is 1.0. For unbounded metrics like MAE or MSE,
        you should set this to a reasonable upper bound on the per-observation loss range.

    """

    def __init__(
        self,
        models,
        metric: metrics.base.Metric,
        delta: float = 0.05,
        loss_range: float = 1.0,
    ):
        super().__init__(models=models, metric=metric)  # type: ignore
        self.delta = delta
        self.loss_range = loss_range

        n = len(models)
        self._metrics = [metric.clone() for _ in range(n)]
        self._active = list(range(n))
        self._n_iterations = 0
        self._best_model_idx = 0

    @abc.abstractmethod
    def _pred_func(self, model): ...

    @property
    def best_model(self):
        """The current best model."""
        return self.models[self._best_model_idx]

    def _hoeffding_epsilon(self, n: int) -> float:
        """Compute the Hoeffding bound epsilon for n observations.

        Uses the corrected bound from the paper that accounts for multiple
        comparisons across all models and all iterations:

            epsilon = sqrt(B^2 * (log(2*n*m) - log(delta)) / n)

        """
        if n == 0:
            return float("inf")
        m = len(self.models)
        return self.loss_range * math.sqrt((math.log(2 * n * m) - math.log(self.delta)) / n)

    def learn_one(self, x, y):
        self._n_iterations += 1

        # Update all active models
        for i in self._active:
            model = self.models[i]
            y_pred = self._pred_func(model)(x)
            self._metrics[i].update(y_true=y, y_pred=y_pred)
            model.learn_one(x, y)

        # Find the best active model
        best_idx = self._active[0]
        for i in self._active[1:]:
            if self._metrics[i].is_better_than(self._metrics[best_idx]):
                best_idx = i
        self._best_model_idx = best_idx

        # Eliminate models using the Hoeffding bound
        if len(self._active) > 1:
            eps = self._hoeffding_epsilon(self._n_iterations)
            best_val = self._metrics[best_idx].get()
            bigger_is_better = self.metric.bigger_is_better

            surviving = []
            for i in self._active:
                if i == best_idx:
                    surviving.append(i)
                    continue
                val = self._metrics[i].get()
                # Check if this model's best possible performance is still worse
                # than the best model's worst possible performance.
                # For "bigger is better" metrics (e.g. accuracy):
                #   eliminate if (val + eps) < (best_val - eps)
                # For "smaller is better" metrics (e.g. MAE):
                #   eliminate if (val - eps) > (best_val + eps)
                if bigger_is_better:
                    if val + eps < best_val - eps:
                        continue  # eliminated
                else:
                    if val - eps > best_val + eps:
                        continue  # eliminated
                surviving.append(i)

            self._active = surviving

    @property
    def n_active_models(self) -> int:
        """The number of models still in the race."""
        return len(self._active)

    @property
    def active_models(self) -> list:
        """The models still in the race."""
        return [self.models[i] for i in self._active]


class HoeffdingRaceRegressor(HoeffdingRace, ModelSelectionRegressor):
    r"""Hoeffding Race model selection for regression.

    Uses the Hoeffding bound to progressively eliminate underperforming models with
    statistical confidence. All active models are trained on each observation. After
    each observation, models whose performance is statistically worse than the current
    best are eliminated from the race.

    This is more efficient than greedy selection when there are many candidate models,
    because eliminated models no longer consume computation. It is more principled than
    successive halving because elimination is based on statistical evidence rather than
    a fixed schedule.

    Parameters
    ----------
    models
        The models to race against each other.
    metric
        Metric used for comparing models. Defaults to `metrics.MAE`.
    delta
        Confidence parameter. Smaller values make elimination harder (more conservative).
        The algorithm is correct with probability at least `1 - delta`. Defaults to 0.05.
    loss_range
        The range of the per-observation loss values. For unbounded metrics you should
        set this to a reasonable upper bound. Defaults to 1.0.

    Examples
    --------

    >>> from river import datasets
    >>> from river import evaluate
    >>> from river import linear_model
    >>> from river import metrics
    >>> from river import model_selection
    >>> from river import optim
    >>> from river import preprocessing

    >>> models = [
    ...     linear_model.LinearRegression(optimizer=optim.SGD(lr=lr))
    ...     for lr in [1e-5, 1e-4, 1e-3, 1e-2]
    ... ]

    >>> dataset = datasets.TrumpApproval()
    >>> metric = metrics.MAE()
    >>> model = (
    ...     preprocessing.StandardScaler() |
    ...     model_selection.HoeffdingRaceRegressor(
    ...         models,
    ...         metric=metric,
    ...         delta=0.05,
    ...         loss_range=50.0,
    ...     )
    ... )

    >>> evaluate.progressive_val_score(dataset, model, metric)
    MAE: 1.319678

    You can inspect how many models remain in the race:

    >>> model["HoeffdingRaceRegressor"].n_active_models
    4

    References
    ----------
    [^1]: [Maron, O. and Moore, A.W., 1993. Hoeffding Races: Accelerating Model Selection Search for Classification and Function Approximation. Advances in Neural Information Processing Systems, 6.](https://papers.nips.cc/paper/1993/hash/02a32ad2669e6fe298e607fe7cc0e1a0-Abstract.html)

    """

    def __init__(
        self,
        models: list,
        metric: metrics.base.RegressionMetric | None = None,
        delta: float = 0.05,
        loss_range: float = 1.0,
    ):
        if metric is None:
            metric = metrics.MAE()
        super().__init__(models=models, metric=metric, delta=delta, loss_range=loss_range)

    def _pred_func(self, model):
        return model.predict_one

    @classmethod
    def _unit_test_params(cls):
        for params in super()._unit_test_params():
            yield {**params, "delta": 0.05, "loss_range": 50.0}


class HoeffdingRaceClassifier(HoeffdingRace, ModelSelectionClassifier):
    r"""Hoeffding Race model selection for classification.

    Uses the Hoeffding bound to progressively eliminate underperforming models with
    statistical confidence. All active models are trained on each observation. After
    each observation, models whose performance is statistically worse than the current
    best are eliminated from the race.

    This is more efficient than greedy selection when there are many candidate models,
    because eliminated models no longer consume computation. It is more principled than
    successive halving because elimination is based on statistical evidence rather than
    a fixed schedule.

    Parameters
    ----------
    models
        The models to race against each other.
    metric
        Metric used for comparing models. Defaults to `metrics.Accuracy`.
    delta
        Confidence parameter. Smaller values make elimination harder (more conservative).
        The algorithm is correct with probability at least `1 - delta`. Defaults to 0.05.
    loss_range
        The range of the per-observation loss values. For accuracy and other metrics
        bounded in [0, 1], this is 1.0. Defaults to 1.0.

    Examples
    --------

    >>> from river import datasets
    >>> from river import evaluate
    >>> from river import linear_model
    >>> from river import metrics
    >>> from river import model_selection
    >>> from river import optim
    >>> from river import preprocessing

    >>> models = [
    ...     linear_model.LogisticRegression(optimizer=optim.SGD(lr=lr))
    ...     for lr in [1e-5, 1e-4, 1e-3, 1e-2]
    ... ]

    >>> dataset = datasets.Phishing()
    >>> model = (
    ...     preprocessing.StandardScaler() |
    ...     model_selection.HoeffdingRaceClassifier(
    ...         models,
    ...         metric=metrics.Accuracy(),
    ...         delta=0.05,
    ...     )
    ... )
    >>> metric = metrics.Accuracy()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    Accuracy: 88.96%

    You can inspect how many models remain in the race:

    >>> model["HoeffdingRaceClassifier"].n_active_models
    2

    References
    ----------
    [^1]: [Maron, O. and Moore, A.W., 1993. Hoeffding Races: Accelerating Model Selection Search for Classification and Function Approximation. Advances in Neural Information Processing Systems, 6.](https://papers.nips.cc/paper/1993/hash/02a32ad2669e6fe298e607fe7cc0e1a0-Abstract.html)

    """

    def __init__(
        self,
        models: list,
        metric: metrics.base.ClassificationMetric | None = None,
        delta: float = 0.05,
        loss_range: float = 1.0,
    ):
        if metric is None:
            metric = metrics.Accuracy()
        super().__init__(models=models, metric=metric, delta=delta, loss_range=loss_range)

    def _pred_func(self, model):
        if self.metric.requires_labels:
            return model.predict_one
        return model.predict_proba_one

    @property
    def _multiclass(self):
        return all(model._multiclass for model in self.models)
