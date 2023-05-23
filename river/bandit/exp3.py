from __future__ import annotations

import math

from river import bandit, proba


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
    burn_in
        The number of steps to use for the burn-in phase. Each arm is given the chance to be pulled
        during the burn-in phase. This is useful to mitigate selection bias.

    References
    ----------
    [^1]: [Auer, P., Cesa-Bianchi, N., Freund, Y. and Schapire, R.E., 2002. The nonstochastic multiarmed bandit problem. SIAM journal on computing, 32(1), pp.48-77.](https://cseweb.ucsd.edu/~yfreund/papers/bandits.pdf)
    [^2]: [Adversarial Bandits and the Exp3 Algorithm — Jeremy Kun](https://jeremykun.com/2013/11/08/adversarial-bandits-and-the-exp3-algorithm/)

    """

    def __init__(self, gamma: float, reward_obj=None, burn_in=0):
        super().__init__(reward_obj, burn_in)
        self.gamma = gamma
