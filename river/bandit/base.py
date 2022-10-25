import abc
import collections
from typing import Counter, DefaultDict, Iterator, List, Union

from river import base, metrics, proba, stats, utils

__all__ = ["ArmID", "Policy", "RewardObj"]

ArmID = Union[int, str]
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

    """

    def __init__(self, reward_obj: RewardObj = None, burn_in=0):
        self.reward_obj = reward_obj or stats.Mean()
        self.burn_in = burn_in
        self._rewards: DefaultDict[ArmID, RewardObj] = collections.defaultdict(
            self.reward_obj.clone
        )
        self._n = 0
        self._counts: Counter = collections.Counter()

    @abc.abstractmethod
    def _pull(self, arm_ids: List[ArmID]) -> ArmID:
        ...

    def pull(self, arm_ids: List[ArmID]) -> Iterator[ArmID]:
        """Pull arm(s).

        This method is a generator that yields the arm(s) that should be pulled. During the burn-in
        phase, all the arms that have not been pulled enough are yielded. Once the burn-in phase is
        over, the policy is allowed to choose the arm(s) that should be pulled. If you only want to
        pull one arm at a time during the burn-in phase, simply call `next(policy.pull(arms))`.

        Parameters
        ----------
        arm_ids
            The list of arms that can be pulled.

        """
        for arm_id in arm_ids:
            if self._counts[arm_id] < self.burn_in:
                yield arm_id
        yield self._pull(arm_ids)

    def update(self, arm_id, *reward_args, **reward_kwargs):
        """Update an arm's state.

        Parameters
        ----------
        arm_id
            The arm to update.
        reward_args
            Positional arguments to pass to the reward object.
        reward_kwargs
            Keyword arguments to pass to the reward object.

        """
        self._rewards[arm_id].update(*reward_args, **reward_kwargs)
        self._counts[arm_id] += 1
        self._n += 1
        return self

    @property
    def ranking(self) -> List[ArmID]:
        """Return the list of arms in descending order of performance."""
        return sorted(
            self._rewards,
            key=lambda arm_id: self._rewards[arm_id],
            reverse=True,
        )

    def __repr__(self):
        ranking = self.ranking
        return utils.pretty.print_table(
            headers=[
                "Arm ID",
                "Reward",
                "Pulls",
                "Share",
            ],
            columns=[
                list(map(str, ranking)),
                [str(self._rewards[arm_id]) for arm_id in ranking],
                [f"{self._counts[arm_id]:,d}" for arm_id in ranking],
                [f"{self._counts[arm_id] / self._n:.2%}" for arm_id in ranking],
            ],
        )
