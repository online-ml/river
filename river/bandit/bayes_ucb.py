from __future__ import annotations

import collections
import random

import scipy.special

from river import bandit, proba


class BayesUCB(bandit.base.Policy):
    """Bayes-UCB bandit policy.

    Bayes-UCB is a Bayesian algorithm for the multi-armed bandit problem. It uses the posterior
    distribution of the reward of each arm to compute an upper confidence bound (UCB) on the
    expected reward of each arm. The arm with the highest UCB is then pulled. The posterior
    distribution is updated after each pull. The algorithm is described in [^1].

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

    >>> policy = bandit.BayesUCB(seed=123)

    >>> metric = stats.Sum()
    >>> while True:
    ...     action = policy.pull(range(env.action_space.n))
    ...     observation, reward, terminated, truncated, info = env.step(action)
    ...     policy = policy.update(action, reward)
    ...     metric = metric.update(reward)
    ...     if terminated or truncated:
    ...         break

    >>> metric
    Sum: 841.

    Reference
    ---------
    [^1]: [Kaufmann, Emilie, Olivier Cappé, and Aurélien Garivier. "On Bayesian upper confidence bounds for bandit problems." Artificial intelligence and statistics. PMLR, 2012.](http://proceedings.mlr.press/v22/kaufmann12/kaufmann12.pdf)

    """

    _REQUIRES_UNIVARIATE_REWARD = True

    def __init__(self, reward_obj=None, burn_in=0, seed: int | None = None):
        super().__init__(reward_obj, burn_in)
        self._posteriors: collections.defaultdict[
            bandit.base.ArmID, proba.Beta
        ] = collections.defaultdict(proba.Beta)
        self.seed = seed
        self._rng = random.Random(seed)

    def _pull(self, arm_ids):
        indices = {arm_id: self.compute_index(arm_id) for arm_id in arm_ids}
        max_index = max(indices.values())
        best_arms = [arm for arm, index in indices.items() if index == max_index]
        return self._rng.choice(best_arms)

    def compute_index(self, arm_id):
        """the p-th quantile of the beta distribution for the arm"""
        p = 1 - 1 / (self._n + 1)
        posterior = self._posteriors[arm_id]
        return scipy.special.btdtri(posterior.alpha, posterior.beta, p)

    def update(self, arm_id, *reward_args, **reward_kwargs):
        """Rewrite update function"""
        super().update(arm_id, *reward_args, **reward_kwargs)
        reward = reward_args[0]
        self._posteriors[arm_id].update(reward)
        return self
