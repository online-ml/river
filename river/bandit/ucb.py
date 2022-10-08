import math

from river import bandit, proba


class UCB(bandit.base.Policy):
    """Upper Confidence Bound (UCB) bandit policy.

    Due to the nature of this algorithm, it's recommended to scale the target so that it exhibits
    sub-gaussian properties. This can be done by using a `preprocessing.TargetStandardScaler`.

    Parameters
    ----------
    delta
        The confidence level.
    reward_obj
        The reward object used to measure the performance of each arm. This can be a metric, a
        statistic, or a distribution.
    burn_in
        The number of steps to use for the burn-in phase. Each arm is given the chance to be pulled
        during the burn-in phase. This is useful to mitigate selection bias.
    seed
        Random number generator seed for reproducibility.

    Examples
    --------

    >>> import gym
    >>> from river import bandit
    >>> from river import stats

    >>> env = gym.make(
    ...     'river_bandits/CandyCaneContest-v0'
    ... )
    >>> _ = env.reset(seed=42)
    >>> _ = env.action_space.seed(123)

    >>> policy = bandit.UCB(delta=100)

    >>> metric = stats.Sum()
    >>> while True:
    ...     action = next(policy.pull(range(env.action_space.n)))
    ...     observation, reward, terminated, truncated, info = env.step(action)
    ...     policy = policy.update(action, reward)
    ...     metric = metric.update(reward)
    ...     if terminated or truncated:
    ...         break

    >>> metric
    Sum: 741.

    References
    ----------
    [^1]: [Lai, T. L., & Robbins, H. (1985). Asymptotically efficient adaptive allocation rules. Advances in applied mathematics, 6(1), 4-22.](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.674.1620&rep=rep1&type=pdf)
    [^2]: [Upper Confidence Bounds - The Multi-Armed Bandit Problem and Its Solutions - Lilian Weng](https://lilianweng.github.io/lil-log/2018/01/23/the-multi-armed-bandit-problem-and-its-solutions.html#upper-confidence-bounds)
    [^3]: [The Upper Confidence Bound Algorithm - Bandit Algorithms](https://banditalgs.com/2016/09/18/the-upper-confidence-bound-algorithm/)

    """

    def __init__(self, delta: float, reward_obj=None, burn_in=0):
        super().__init__(reward_obj, burn_in)
        self.delta = delta

    def _pull(self, arms):
        upper_bounds = {
            arm: (
                reward.mode
                if isinstance(reward, proba.base.Distribution)
                else reward.get()
                + self.delta * math.sqrt(2 * math.log(self._n) / self._counts[arm])
            )
            if (reward := self._rewards.get(arm)) is not None
            else math.inf
            for arm in arms
        }
        return max(arms, key=lambda arm: upper_bounds[arm])

    @classmethod
    def _unit_test_params(cls):
        yield {"delta": 1}
