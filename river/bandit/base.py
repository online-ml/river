import abc
import collections
import random
from typing import Hashable, Iterator, List, Union
from river import base
from river import metrics
from river import stats

Arm = Hashable
RewardObj = Union[stats.base.Statistic, metrics.base.Metric]

class BanditPolicy(base.Base, abc.ABC):

    def __init__(self, reward_obj: RewardObj = None, seed: int = None):
        self.reward_obj = reward_obj or stats.Sum()
        self.seed = seed
        self.rng = random.Random(seed)
        self._rewards: typing.DefaultDict[Arm, RewardObj] = collections.defaultdict(self.reward_obj.clone)
        self._n = 0
        self._best_arm: Arm = None

    @property
    def best_arm(self):
        return self._best_arm

    @abc.abstractmethod
    def pull(self, arms: List[Arm]) -> Iterator[Arm]:
        ...

    def update(self, arm, *reward_args, **reward_kwargs):
        self._rewards[arm].update(*reward_args, **reward_kwargs)
        # The > operator assumes the reward object is a metric or a statistic
        if self._best_arm is None or self._rewards[arm] > self._rewards[self._best_arm]:
            self._best_arm = arm
