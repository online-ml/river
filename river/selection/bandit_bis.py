from abc import ABC, abstractmethod
from dataclasses import dataclass
from random import Random

from river.metrics import Metric

from .base import ModelSelectionRegressor


@dataclass
class Arm:
    """

    An arm may also be referred to as a lever.

    """

    index: int
    metric: Metric

    def __gt__(self, other):
        if self.metric.bigger_is_better:
            return self.metric.get() > other.metric.get()
        return self.metric.get() < other.metric.get()


class MAB(ABC):
    """Multi-armed bandit (MAB).

    A multi-armed bandit is composed of $k$ arms.

    """

    def __init__(self, n_arms, seed):
        self.arms = [Arm() for _ in range(n_arms)]
        self.seed = seed
        self.rng = Random(seed)
        self.best_arm = self.arms[0]

    @abstractmethod
    def pull(self) -> Arm:
        ...

    def update(self, arm: Arm, reward: float):
        arm.n_pulls += 1
        arm.reward_sum += reward
        if arm.index != self.best_arm.index and arm > self.best_arm:
            self.best_arm = arm


class EpsilonGreedy(MAB):
    def __init__(self, eps, k, seed=None):
        super().__init__(k, seed)
        self.eps = eps

    def pull(self):
        if self.rng.random() > self.eps:
            return self.best_arm
        return self.rng.choice(self.arms)


class EpsilonGreedyRegressor(ModelSelectionRegressor):
    def __init__(self, models, metric, eps, seed=None):
        super().__init__(models, metric)
        self._mab = EpsilonGreedy(eps=eps, k=len(models), seed=None)

    @property
    def eps(self):
        return self._mab.eps

    @property
    def seed(self):
        return self._mab.seed

    def learn_one(self, x, y):
        arm = self._mab.pull()
        model = self[arm.index]
        y_pred = model.predict_one(x)
        self._mab.update(
            arm,
        )
        model.learn_one(x, y)
        return self
