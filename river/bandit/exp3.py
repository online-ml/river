from __future__ import annotations

import collections
import functools
import math
import random

from river import bandit


class Exp3(bandit.base.Policy):
    """Exp3 bandit policy.

    This policy works by maintaining a weight for each arm. These weights are used to randomly
    decide which arm to pull. The weights are increased or decreased, depending on the reward. An
    egalitarianism factor $\\gamma \\in [0, 1]$ is included, to tune the desire to pick an arm
    uniformly at random. That is, if $\\gamma = 1$, the arms are picked uniformly at random.

    Parameters
    ----------
    gamma
        The egalitarianism factor. Setting this to 0 leads to what is called the EXP3 policy.
    reward_obj
        The reward object used to measure the performance of each arm. This can be a metric, a
        statistic, or a distribution.
    reward_scaler
        A reward scaler used to scale the rewards before they are fed to the reward object. This
        can be useful to scale the rewards to a (0, 1) range for instance.
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

    >>> policy = bandit.Exp3(gamma=0.5, seed=42)

    >>> metric = stats.Sum()
    >>> while True:
    ...     action = policy.pull(range(env.action_space.n))
    ...     observation, reward, terminated, truncated, info = env.step(action)
    ...     policy = policy.update(action, reward)
    ...     metric = metric.update(reward)
    ...     if terminated or truncated:
    ...         break

    >>> metric
    Sum: 799.

    References
    ----------
    [^1]: [Auer, P., Cesa-Bianchi, N., Freund, Y. and Schapire, R.E., 2002. The nonstochastic multiarmed bandit problem. SIAM journal on computing, 32(1), pp.48-77.](https://cseweb.ucsd.edu/~yfreund/papers/bandits.pdf)
    [^2]: [Adversarial Bandits and the Exp3 Algorithm — Jeremy Kun](https://jeremykun.com/2013/11/08/adversarial-bandits-and-the-exp3-algorithm/)

    """

    _REQUIRES_UNIVARIATE_REWARD = True

    def __init__(
        self, gamma: float, reward_obj=None, reward_scaler=None, burn_in=0, seed: int | None = None
    ):
        super().__init__(reward_obj=reward_obj, reward_scaler=reward_scaler, burn_in=burn_in)
        self.seed = seed
        self.gamma = gamma
        self._rng = random.Random(seed)
        self._weights: collections.defaultdict = collections.defaultdict(
            functools.partial(float, 1)
        )
        self._probabilities: dict[bandit.base.ArmID, float] = {}

    def _pull(self, arm_ids):
        total = sum(self._weights[arm_id] for arm_id in arm_ids)
        self._probabilities = {
            arm_id: (1 - self.gamma) * (self._weights[arm_id] / total) + self.gamma / len(arm_ids)
            for arm_id in arm_ids
        }
        return self._rng.choices(arm_ids, weights=self._probabilities.values())[0]

    def update(self, arm_id, *reward_args, **reward_kwargs):
        super().update(arm_id, *reward_args, **reward_kwargs)
        reward = reward_args[0]
        reward /= self._probabilities[arm_id]
        self._weights[arm_id] *= math.exp(self.gamma * reward / len(self._weights))
        return self

    @classmethod
    def _unit_test_params(cls):
        yield {"gamma": 0}
        yield {"gamma": 0.1}
        yield {"gamma": 0.5}
        yield {"gamma": 0.9}
        yield {"gamma": 1}
