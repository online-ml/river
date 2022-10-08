from river import bandit
from river import proba


class ThompsonSampling(bandit.base.Policy):
    """Thompson sampling.

    Thompson sampling is often used with a Beta distribution. However, any probability distribution
    can be used, as long it makes sense with the reward shape. For instance, a Beta distribution
    is meant to be used with binary rewards, while a Gaussian distribution is meant to be used with
    continuous rewards.

    Parameters
    ----------
    dist
        A distribution to sample from.
    burn_in
        The number of steps to use for the burn-in phase. Each arm is given the chance to be pulled
        during the burn-in phase. This is useful to mitigate selection bias.
    seed
        Random number generator seed for reproducibility.

    References
    ----------
    [^1]: [An Empirical Evaluation of Thompson Sampling](https://proceedings.neurips.cc/paper/2011/file/e53a0a2978c28872a4505bdb51db06dc-Paper.pdf)

    """

    def __init__(self, dist: proba.base.Distribution, burn_in=0, seed=None):
        super().__init__(dist, burn_in, seed)

    @property
    def dist(self):
        return self.reward_obj

    def _pull(self, arms):
        return max(arms, key=lambda arm: self._rewards[arm].sample())

    @classmethod
    def _unit_test_params(cls):
        yield {"dist": proba.Gaussian()}
