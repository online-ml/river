from __future__ import annotations

from river import bandit, metrics, model_selection


class BanditRegressor(model_selection.base.ModelSelectionRegressor):
    """Bandit-based model selection.

    Each model is associated with an arm. At each `learn_one` call, the policy decides which
    arm/model to pull. The reward is the performance of the model on the provided sample. The
    `predict_one` method uses the current best model.

    Parameters
    ----------
    models
        The models to select from.
    metric
        The metric that is used to measure the performance of each model.
    policy
        The bandit policy to use.

    Examples
    --------

    >>> from river import bandit
    >>> from river import datasets
    >>> from river import evaluate
    >>> from river import linear_model
    >>> from river import metrics
    >>> from river import model_selection
    >>> from river import optim
    >>> from river import preprocessing

    >>> models = [
    ...     linear_model.LinearRegression(optimizer=optim.SGD(lr=lr))
    ...     for lr in [0.0001, 0.001, 1e-05, 0.01]
    ... ]

    >>> dataset = datasets.TrumpApproval()
    >>> model = (
    ...     preprocessing.StandardScaler() |
    ...     model_selection.BanditRegressor(
    ...         models,
    ...         metric=metrics.MAE(),
    ...         policy=bandit.EpsilonGreedy(
    ...             epsilon=0.1,
    ...             decay=0.001,
    ...             burn_in=100,
    ...             seed=42
    ...         )
    ...     )
    ... )
    >>> metric = metrics.MAE()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    MAE: 3.13815

    Here's another example using the UCB policy. The latter is more sensitive to the target scale,
    and usually works better when the target is rescaled.

    >>> models = [
    ...     linear_model.LinearRegression(optimizer=optim.SGD(lr=lr))
    ...     for lr in [0.0001, 0.001, 1e-05, 0.01]
    ... ]

    >>> model = (
    ...     preprocessing.StandardScaler() |
    ...     preprocessing.TargetStandardScaler(
    ...         model_selection.BanditRegressor(
    ...             models,
    ...             metric=metrics.MAE(),
    ...             policy=bandit.UCB(
    ...                 delta=1,
    ...                 burn_in=100
    ...             )
    ...         )
    ...     )
    ... )
    >>> metric = metrics.MAE()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    MAE: 0.875457

    """

    def __init__(
        self,
        models,
        metric: metrics.base.RegressionMetric,
        policy: bandit.base.Policy,
    ):
        super().__init__(models, metric)
        self.policy = policy.clone({"reward_obj": self.metric})

    @property
    def best_model(self):
        if not self.policy._rewards:
            return None
        best_arm_id = max(self.policy._rewards, key=lambda arm: self.policy._rewards[arm].get())
        return self[best_arm_id]

    def learn_one(self, x, y):
        # Update every model as long as it has not been pulled enough times
        # This provides a significant accuracy boost
        arm_ids = [
            arm_id
            for arm_id in range(len(self.models))
            if self.policy._counts[arm_id] < self.policy.burn_in
        ]
        if not arm_ids:
            arm_ids = [self.policy.pull(range(len(self.models)))]

        for arm_id in arm_ids:
            model = self[arm_id]
            y_pred = model.predict_one(x)
            self.policy.update(arm_id, y_true=y, y_pred=y_pred)
            model.learn_one(x, y)

        return self

    @classmethod
    def _unit_test_params(cls):
        for params in super()._unit_test_params():
            yield {**params, "policy": bandit.EpsilonGreedy(0.2)}

    def _unit_test_skips(self):
        return (
            {"check_model_selection_order_does_not_matter"} if hasattr(self.policy, "seed") else {}
        )
