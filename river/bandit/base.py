from __future__ import annotations

import abc
import collections
import typing
from collections import Counter

from river import base, compose, metrics, proba, stats, utils

__all__ = ["ArmID", "Policy", "ContextualPolicy", "RewardObj"]

ArmID = typing.Union[int, str]  # noqa: UP007
RewardObj = typing.Union[  # noqa: UP007
    stats.base.Statistic, metrics.base.Metric, proba.base.Distribution
]


class Policy(base.Base, abc.ABC):
    """Bandit policy base class.

    Parameters
    ----------
    reward_obj
        The reward object used to measure the performance of each arm. This can be a metric, a
        statistic, or a distribution.
    reward_scaler
        A reward scaler used to scale the rewards before they are fed to the reward object. This
        can be useful to scale the rewards to a (0, 1) range for instance.
    burn_in
        The number of steps to use for the burn-in phase. Each arm is given the chance to be pulled
        during the burn-in phase. This is useful to mitigate selection bias.

    """

    _REQUIRES_UNIVARIATE_REWARD = False

    def __init__(
        self,
        reward_obj: RewardObj | None = None,
        reward_scaler: compose.TargetTransformRegressor | None = None,
        burn_in=0,
    ):
        self.reward_obj = reward_obj or stats.Mean()
        self.reward_scaler = reward_scaler
        self.burn_in = burn_in
        self._rewards: collections.defaultdict[ArmID, RewardObj] = collections.defaultdict(
            self.reward_obj.clone
        )
        self._n = 0
        self._counts: Counter = collections.Counter()

    def __post_init__(self):
        # It's only possible to use a reward scaler if the reward object is updated with univariate
        # reward values, because it manipulates the reward values directly.
        if self._REQUIRES_UNIVARIATE_REWARD or (
            self.reward_scaler
            and not (
                isinstance(self.reward_obj, proba.base.Distribution)
                or isinstance(self.reward_obj, stats.base.Univariate)
            )
        ):
            raise ValueError(
                "The reward object should be a distribution or a univariate statistic if a "
                "reward scaler is used."
            )

    @abc.abstractmethod
    def _pull(self, arm_ids: list[ArmID]) -> ArmID:
        ...

    def pull(self, arm_ids: list[ArmID]) -> ArmID:
        """Pull arm(s).

        This method is a generator that yields the arm(s) that should be pulled. During the burn-in
        phase, all the arms that have not been pulled enough times are yielded. Once the burn-in
        phase is over, the policy is allowed to choose the arm(s) that should be pulled. If you
        only want to pull one arm at a time during the burn-in phase, simply call
        `next(policy.pull(arms))`.

        Parameters
        ----------
        arm_ids
            The list of arms that can be pulled.

        Returns
        -------
        A single arm.

        """
        for arm_id in arm_ids:
            if self._counts[arm_id] < self.burn_in:
                return arm_id
        return self._pull(arm_ids)

    def update(self, arm_id, *reward_args, **reward_kwargs):
        """Update an arm's state.

        Parameters
        ----------
        arm_id
            The arm to update.
        reward_kwargs
            Keyword arguments to pass to the reward object.

        """

        if self.reward_scaler:
            reward = reward_args[0]
            self.reward_scaler._update(y=reward)
            reward = self.reward_scaler.func(reward)
            reward_args = (reward,)

        self._rewards[arm_id].update(*reward_args, **reward_kwargs)
        self._counts[arm_id] += 1
        self._n += 1
        return self

    @property
    def ranking(self) -> list[ArmID]:
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
                [f"{(self._counts[arm_id] / self._n) if self._n else 0:.2%}" for arm_id in ranking],
            ],
        )


class ContextualPolicy(Policy):
    """Contextual bandit policy base class.

    Parameters
    ----------
    reward_obj
        The reward object used to measure the performance of each arm. This can be a metric, a
        statistic, or a distribution.
    reward_scaler
        A reward scaler used to scale the rewards before they are fed to the reward object. This
        can be useful to scale the rewards to a (0, 1) range for instance.
    burn_in
        The number of steps to use for the burn-in phase. Each arm is given the chance to be pulled
        during the burn-in phase. This is useful to mitigate selection bias.

    """

    @abc.abstractmethod
    def _pull(self, arm_ids: list[ArmID], context: dict) -> ArmID:  # type: ignore[override]
        ...

    def pull(self, arm_ids: list[ArmID], context: dict = None) -> ArmID:
        """Pull arm(s).

        This method is a generator that yields the arm(s) that should be pulled. During the burn-in
        phase, all the arms that have not been pulled enough times are yielded. Once the burn-in
        phase is over, the policy is allowed to choose the arm(s) that should be pulled. If you
        only want to pull one arm at a time during the burn-in phase, simply call
        `next(policy.pull(arms))`.

        Parameters
        ----------
        arm_ids
            The list of arms that can be pulled.
        context
            The context associated with the arm. Doesn't have to be provided if the policy is not
            contextual.

        Returns
        -------
        A single arm.

        """
        for arm_id in arm_ids:
            if self._counts[arm_id] < self.burn_in:
                return arm_id
        return self._pull(arm_ids, context=context)  # type: ignore[arg-type]

    def update(self, arm_id, context, *reward_args, **reward_kwargs):
        """Update an arm's state.

        Parameters
        ----------
        arm_id
            The arm to update.
        context
            The context associated with the arm. Doesn't have to be provided if the policy is not
            contextual.
        reward_kwargs
            Keyword arguments to pass to the reward object.

        """

        if self.reward_scaler:
            reward = reward_args[0]
            self.reward_scaler._update(y=reward)
            reward = self.reward_scaler.func(reward)
            reward_args = (reward,)

        self._rewards[arm_id].update(*reward_args, **reward_kwargs)
        self._counts[arm_id] += 1
        self._n += 1
        return self
