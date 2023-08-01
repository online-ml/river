from __future__ import annotations

import collections
import random

from river import bandit, proba


class RandomPolicy(bandit.base.Policy):
    """Random bandit policy.

    This policy simply pulls a random arm at each time step. It is useful as a baseline.

    Parameters
    ----------
    reward_obj
        The reward object that is used to update the posterior distribution.
    burn_in
        Number of initial observations per arm before using the posterior distribution.
    seed
        Random number generator seed for reproducibility.

    Examples
    --------

    >>> import gym
    >>> from river import bandit
    >>> from river import proba
    >>> from river import stats

    >>> env = gym.make(
    ...     'river_bandits/CandyCaneContest-v0'
    ... )
    >>> _ = env.reset(seed=42)
    >>> _ = env.action_space.seed(123)

    >>> policy = bandit.RandomPolicy(seed=123)

    >>> metric = stats.Sum()
    >>> while True:
    ...     action = policy.pull(range(env.action_space.n))
    ...     observation, reward, terminated, truncated, info = env.step(action)
    ...     policy = policy.update(action, reward)
    ...     metric = metric.update(reward)
    ...     if terminated or truncated:
    ...         break

    >>> metric
    Sum: 755.

    """

    def __init__(self, reward_obj=None, burn_in=0, seed: int | None = None):
        super().__init__(reward_obj, burn_in)
        self._posteriors: collections.defaultdict[
            bandit.base.ArmID, proba.Beta
        ] = collections.defaultdict(proba.Beta)
        self.seed = seed
        self._rng = random.Random(seed)

    def _pull(self, arm_ids):
        return self._rng.choice(arm_ids)
