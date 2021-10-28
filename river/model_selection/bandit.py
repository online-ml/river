"""

This module contains model selection logic based on multi-armed bandits (MAB). The way the code is
organised, the bandit logic is agnostic of the model selection aspect.

"""
import math
import operator
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from random import Random
from typing import Any, Iterator, List

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


class Bandit:
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

    @property
    def ranking(self):
        return [
            arm.index
            for arm in sorted(
                self.arms,
                key=lambda arm: arm.metric.get(),
                reverse=self.metric.bigger_is_better,
            )
        ]

    def __repr__(self):
        return utils.pretty.print_table(
            headers=[
                "Ranking",
                self.metric.__class__.__name__,
                "Pulls",
                "Share",
            ],
            columns=[
                [str(r) for r in self.ranking],
                [f"{arm.metric.get():{self.metric._fmt}}" for arm in self.arms],
                [f"{arm.n_pulls:,d}" for arm in self.arms],
                [f"{arm.n_pulls / self.n_pulls:.2%}" for arm in self.arms],
            ],
        )


class BanditSolver(ABC):
    """A solver for bandit problems.

    A solver is in charge of solving a bandit problem.

    To use a solver, first call `pull` to pick an arm. This method is specific to each solver.
    Then call `update` to update the arm, as well as determine if the arm is the best arm. This
    method is common to all solvers.

    """

    def __init__(self, burn_in: int, seed: int):
        self.burn_in = burn_in
        self.seed = seed
        self.rng = Random(seed)

    def pull(self, bandit: Bandit) -> Iterator[Arm]:
        burn_in_over = True
        for arm in bandit.arms:
            if arm.n_pulls < self.burn_in:
                yield arm
                burn_in_over = False
        if burn_in_over:
            yield from self._pull(bandit)

    @abstractmethod
    def _pull(self, bandit: Bandit) -> Iterator[Arm]:
        ...


class EpsilonGreedy(BanditSolver):
    r"""$\eps$-greedy strategy."""

    def __init__(self, epsilon: float, decay: float, burn_in, seed):
        super().__init__(burn_in, seed)
        self.epsilon = epsilon
        self.decay = decay

    def current_epsilon(self, n: int):
        if self.decay:
            return self.epsilon * math.exp(-n * self.decay)
        return self.epsilon

    def _pull(self, bandit: Bandit):
        yield (
            self.rng.choice(bandit.arms)  # explore
            if self.rng.random() < self.current_epsilon(n=bandit.n_pulls)
            else bandit.best_arm  # exploit
        )


class BanditRegressor(ModelSelectionRegressor):
    def __init__(
        self, models: List[base.Regressor], bandit: Bandit, solver: BanditSolver
    ):
        super().__init__(models, bandit.metric)
        self.bandit = bandit
        self.solver = solver

    @property
    def best_model(self):
        return self[self.bandit.best_arm.index]

    @property
    def metrics(self):
        return [arm.metric for arm in self.bandit.arms]

    def learn_one(self, x, y):
        for arm in self.solver.pull(self.bandit):
            model = self[arm.index]
            y_pred = model.predict_one(x)
            self.bandit.update(arm, y, y_pred)
            model.learn_one(x, y)
        return self


class EpsilonGreedyRegressor(BanditRegressor):
    r"""Model selection based on the $\eps$-greedy bandit strategy.

    Performs model selection by using an $\eps$-greedy bandit strategy. A model is selected for
    each learning step. The best model is selected (1 - $\eps$%) of the time.

    Selection bias is a common problem when using bandits for online model selection. This bias can
    be mitigated by using a burn-in phase. Each model is given the chance to learn during the first
    `burn_in` steps.

    Parameters
    ----------
    models
        The models to choose from.
    epsilon
        The fraction of time the best model is selected.
    decay
        The rate at which `epsilon` decays.
    metric
        The metric that is used to compare models with each other.
    seed
        Random number generator seed for reproducibility.

    Examples
    --------

    >>> from river import compose
    >>> from river import datasets
    >>> from river import evaluate
    >>> from river import linear_model
    >>> from river import metrics
    >>> from river import model_selection
    >>> from river import optim
    >>> from river import preprocessing

    >>> models = [
    ...     compose.Pipeline(
    ...         preprocessing.StandardScaler(),
    ...         linear_model.LinearRegression(optimizer=optim.SGD(lr=lr))
    ...     )
    ...     for lr in [1e-5, 1e-4, 1e-3, 1e-2]
    ... ]

    >>> dataset = datasets.TrumpApproval()
    >>> selector = model_selection.EpsilonGreedyRegressor(
    ...     models,
    ...     epsilon=0.1,
    ...     decay=0.001,
    ...     burn_in=100,
    ...     seed=1
    ... )
    >>> metric = metrics.MAE()

    >>> evaluate.progressive_val_score(dataset, selector, metric)
    MAE: 1.370979

    >>> selector.bandit
    Ranking   MAE         Pulls   Share
          3   16.197726     111    8.53%
          2   15.022596     117    8.99%
          1   14.036779     109    8.38%
          0    1.425602     964   74.10%

    >>> selector.best_model
    Pipeline (
      StandardScaler (
        with_std=True
      ),
      LinearRegression (
        optimizer=SGD (
          lr=Constant (
            learning_rate=0.01
          )
        )
        loss=Squared ()
        l2=0.
        intercept_init=0.
        intercept_lr=Constant (
          learning_rate=0.01
        )
        clip_gradient=1e+12
        initializer=Zeros ()
      )
    )

    """

    def __init__(
        self,
        models,
        epsilon=0.1,
        decay=0.0,
        burn_in=100,
        metric=None,
        seed: int = None,
    ):
        if metric is None:
            metric = metrics.MAE()
        super().__init__(
            models=models,
            bandit=Bandit(n_arms=len(models), metric=metric),
            solver=EpsilonGreedy(
                epsilon=epsilon,
                decay=decay,
                burn_in=burn_in,
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
    def burn_in(self):
        return self.solver.burn_in

    @property
    def seed(self):
        return self.solver.seed
