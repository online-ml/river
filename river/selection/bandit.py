"""

This module contains model selection logic based on multi-armed bandits (MAB). The way the code is
organised, the bandit logic is agnostic of the model selection aspect.

"""
import math
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from random import Random
from typing import Any, List

from river import base, metrics, utils

from .base import ModelSelectionRegressor


@dataclass
class Arm:
    """An arm in a multi-armed bandit.

    It may also be referred to as a "lever".

    """

    index: int
    metric: metrics.Metric
    n_pulls: int = 0


class Bandit(ABC):
    """(Multi-armed) bandit (MAB) solver.

    A bandit is composed of multiple arms. A solver is in charge of determining the best one.

    """

    def __init__(self, n_arms: int, metric: metrics.Metric):
        self.arms = [Arm(index=i, metric=deepcopy(metric)) for i in range(n_arms)]
        self.metric = metric
        self.best_arm = self.arms[0]
        self.n_pulls = 0

    def update(self, arm: Arm, y_true: Any, y_pred: Any):
        self.n_pulls += 1
        arm.n_pulls += 1
        arm.metric.update(y_true, y_pred)
        # Check for a new best arm
        if arm.index != self.best_arm.index and arm.metric.is_better_than(
            self.best_arm.metric
        ):
            self.best_arm = arm

    def __repr__(self):
        return utils.pretty.print_table(
            headers=["Index", self.metric.__class__.__name__, "Pulls", "Share"],
            columns=[
                [str(arm.index) for arm in self.arms],
                [f"{arm.metric.get():{self.metric._fmt}}" for arm in self.arms],
                [f"{arm.n_pulls:,d}" for arm in self.arms],
                [f"{arm.n_pulls / self.n_pulls:.2%}" for arm in self.arms],
            ],
        )


class BanditSolver:
    """A solver for bandit problems.

    A solver is in charge of solving a bandit problem.

    To use a solver, first call `pull` to pick an arm. This method is specific to each solver.
    Then call `update` to update the arm, as well as determine if the arm is the best arm. This
    method is common to all solvers.

    """

    def __init__(self, bandit, seed: int):
        self.bandit = bandit
        self.seed = seed
        self.rng = Random(seed)

    @abstractmethod
    def pull(self) -> Arm:
        ...


class EpsilonGreedy(BanditSolver):
    r"""$\eps$-greedy strategy."""

    def __init__(self, bandit, epsilon: float, decay=0.0, seed=None):
        super().__init__(bandit, seed)
        self.epsilon = epsilon
        self.decay = decay

    @property
    def current_epsilon(self):
        if self.decay:
            return self.epsilon * math.exp(-self.bandit.n_pulls * self.decay)
        return self.epsilon

    def pull(self):
        return (
            self.rng.choice(self.bandit.arms)  # explore
            if self.rng.random() < self.current_epsilon
            else self.bandit.best_arm  # exploit
        )


class BanditRegressor(ModelSelectionRegressor):
    def __init__(
        self,
        models: List[base.Regressor],
        solver: BanditSolver,
        metric: metrics.RegressionMetric = None,
    ):
        super().__init__(models, metric)
        self.solver = solver

    @property
    def bandit(self):
        return self.solver.bandit

    @property
    def best_model(self):
        return self[self.bandit.best_arm.index]

    def learn_one(self, x, y):
        arm = self.solver.pull()
        model = self[arm.index]
        y_pred = model.predict_one(x)
        self.bandit.update(arm, y, y_pred)
        model.learn_one(x, y)
        return self


class EpsilonGreedyRegressor(BanditRegressor):
    r"""Model selection based on the $\eps$-greedy bandit strategy.

    Parameters
    ----------
    models
    epsilon
    decay
    metric
    seed

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
    >>> selector = EpsilonGreedyRegressor(models, epsilon=0.8, decay=0.01, seed=1)
    >>> metric = metrics.MAE()

    >>> evaluate.progressive_val_score(dataset, selector, metric)
    MAE: 2.797875

    >>> selector.bandit
    Index   MAE              Pulls   Share
        0        32.114043      23    2.30%
        1        28.959042      23    2.30%
        2         1.576562     929   92.81%
        3   623,644.303637      26    2.60%

    >>> selector.best_model["LinearRegression"].optimizer.lr
    Constant({'learning_rate': 0.01})

    """

    def __init__(self, models, epsilon=0.1, decay=0.0, metric=None, seed=None):
        if metric is None:
            metric = metrics.MAE()
        super().__init__(
            models=models,
            metric=metric,
            solver=EpsilonGreedy(
                bandit=Bandit(n_arms=len(models), metric=metric),
                epsilon=epsilon,
                decay=decay,
                seed=seed,
            ),
        )

    @property
    def epsilon(self):
        return self.solver.epsilon

    @property
    def decay(self):
        return self.solver.decay

    @property
    def seed(self):
        return self.solver.seed
