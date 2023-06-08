from __future__ import annotations

import random

from river import bandit, proba


class ThompsonSampling(bandit.base.Policy):
    """Thompson sampling.

    Thompson sampling is often used with a Beta distribution. However, any probability distribution
    can be used, as long it makes sense with the reward shape. For instance, a Beta distribution
    is meant to be used with binary rewards, while a Gaussian distribution is meant to be used with
    continuous rewards.

    The randomness of a distribution is controlled by its seed. The seed should not set within the
    distribution, but should rather be defined in the policy parametrization. In other words, you
    should do this:

    ```python\n
    policy = ThompsonSampling(dist=proba.Beta(1, 1), seed=42)\n
    ```

    and not this:

    ```python\n
    policy = ThompsonSampling(dist=proba.Beta(1, 1, seed=42))\n
    ```

    Parameters
    ----------
    reward_obj
        A distribution to sample from.
    burn_in
        The number of steps to use for the burn-in phase. Each arm is given the chance to be pulled
        during the burn-in phase. This is useful to mitigate selection bias.
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

    >>> policy = bandit.ThompsonSampling(reward_obj=proba.Beta(), seed=101)

    >>> metric = stats.Sum()
    >>> while True:
    ...     arm = policy.pull(range(env.action_space.n))
    ...     observation, reward, terminated, truncated, info = env.step(arm)
    ...     policy = policy.update(arm, reward)
    ...     metric = metric.update(reward)
    ...     if terminated or truncated:
    ...         break

    >>> metric
    Sum: 820.

    References
    ----------
    [^1]: [An Empirical Evaluation of Thompson Sampling](https://proceedings.neurips.cc/paper/2011/file/e53a0a2978c28872a4505bdb51db06dc-Paper.pdf)

    """

    def __init__(
        self, reward_obj: proba.base.Distribution = None, burn_in=0, seed: int | None = None
    ):
        super().__init__(reward_obj=reward_obj, burn_in=burn_in)
        self.seed = seed
        self._rng = random.Random(seed)
        self._rewards.default_factory = self._clone_reward_obj_with_seed

    def _clone_reward_obj_with_seed(self):
        return self.reward_obj.clone({"seed": self._rng.randint(0, 2**32)})

    def _pull(self, arm_ids):
        return max(arm_ids, key=lambda arm_id: self._rewards[arm_id].sample())

    @classmethod
    def _unit_test_params(cls):
        yield {"reward_obj": proba.Beta()}
