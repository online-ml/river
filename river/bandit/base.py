import abc
import collections
import random
from typing import Hashable, Iterator, List, Union
from river import base
from river import metrics
from river import proba
from river import stats
from river import utils

__all__ = ["Arm", "Policy", "RewardObj"]

Arm = Hashable
RewardObj = Union[stats.base.Statistic, metrics.base.Metric, proba.base.Distribution]

class Policy(base.Base, abc.ABC):
    """Bandit policy base class.

    Parameters
    ----------
    reward_obj
        The reward object used to measure the performance of each arm. This can be a metric, a
        statistic, or a distribution.
    burn_in
        The number of steps to use for the burn-in phase. Each arm is given the chance to be pulled
        during the burn-in phase. This is useful to mitigate selection bias.
    seed
        Random number generator seed for reproducibility.

    """

    def __init__(self, reward_obj: RewardObj = None, burn_in=0, seed: int = None):
        self.reward_obj = reward_obj or stats.Sum()
        self.burn_in = burn_in
        self.seed = seed
        self.rng = random.Random(seed)
        self._rewards: typing.DefaultDict[Arm, RewardObj] = collections.defaultdict(self.reward_obj.clone)
        self._n = 0
        self._best_arm: Arm = None
        self._counts = collections.Counter()

    @property
    def best_arm(self):
        """The best arm currently."""
        return self._best_arm

    @abc.abstractmethod
    def _pull(self, arms: List[Arm]) -> Arm:
        ...

    def pull(self, arms: List[Arm]) -> Iterator[Arm]:
        """Pull arm(s).

        This method is a generator that yields the arm(s) that should be pulled. During the burn-in
        phase, all the arms that have not been pulled enough are yielded. Once the burn-in phase is
        over, the policy is allowed to choose the arm(s) that should be pulled. If you only want to
        pull one arm at a time during the burn-in phase, simply call `next(policy.pull(arms))`.

        Parameters
        ----------
        arms
            The list of arms that can be pulled.

        """
        for arm in arms:
            if self._counts[arm] < self.burn_in:
                yield arm
        yield self._pull(arms)

    def update(self, arm, *reward_args, **reward_kwargs):
        """Update an arm's state.

        Parameters
        ----------
        arm
            The arm to update.
        reward_args
            Positional arguments to pass to the reward object.
        reward_kwargs
            Keyword arguments to pass to the reward object.

        """
        self._rewards[arm].update(*reward_args, **reward_kwargs)
        self._counts[arm] += 1
        self._n += 1
        for arm, reward in self._rewards.items():
            # The > operator assumes the reward object is a metric, a statistic, or a distribution
            if self._best_arm is None or reward > self._rewards[self._best_arm]:
                self._best_arm = arm

    @property
    def ranking(self) -> List[Arm]:
        """Return the list of arms in descending order of performance."""
        return sorted(
            self._rewards,
            key=lambda arm: self._rewards[arm],
            reverse=True,
        )

    def __repr__(self):
        ranking = self.ranking
        return utils.pretty.print_table(
            headers=[
                "Arm",
                "Reward",
                "Pulls",
                "Share",
            ],
            columns=[
                list(map(str, ranking)),
                [str(self._rewards[arm]) for arm in ranking],
                [f"{self._counts[arm]:,d}" for arm in ranking],
                [f"{self._counts[arm] / self._n:.2%}" for arm in ranking],
            ],
        )
