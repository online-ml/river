from __future__ import annotations

import random

from river import bandit
from river import proba

from scipy import special



class BayesUCB(bandit.base.Policy):
    """Bayes-UCB bandit policy.


    Parameters
    --------
    n_arms : int. Number of arms.
    posterior : dict. Contains the posterior distributions of all the arms.
    reward_obj : The reward object used to measure the performance of each arm. This can be a metric, a statistic, or a distribution.
    burn_in : The number of steps to use for the burn-in phase. Each arm is given the chance to be pulled during the burn-in phase. This is useful to mitigate selection bias.

    Examples
    --------
    
    >>> env = gym.make('river_bandits/CandyCaneContest-v0')

    >>> _ = env.reset(seed=42)
    >>> _ = env.action_space.seed(123)

    >>> policy = BayesUCB(n_arms=env.action_space.n)

    >>> metric = stats.Sum()  # cumulative reward

    >>> while True:
    ...    action = next(policy.pull(range(env.action_space.n)))
    ...    observation, reward, terminated, truncated, info = env.step(action)
    ...    policy = policy.update(action, reward)
    ...    policy.get_reward(action, reward)
    ...    metric = metric.update(reward)
    ...    if terminated or truncated:
    ...        break

    >>> metric
    Sum: 824.

    Reference
    --------
    [1] Kaufmann, Emilie, Olivier Cappé, and Aurélien Garivier. "On Bayesian upper confidence bounds for bandit problems." Artificial intelligence and statistics. PMLR, 2012.  
    """

    def __init__(self, n_arms: int, reward_obj=None, burn_in=0):
        super().__init__(reward_obj, burn_in)
        self.n_arms = n_arms
        self.posterior = dict()
        for arm_id in range(self.n_arms):
            self.posterior[arm_id] = proba.Beta()


    def _pull(self, arm_ids):
        index = dict()
        for arm_id in arm_ids:
            index[arm_id] = self.compute_index(arm_id)
        max_index = max(index.values())

        best_arms = [arm for arm in index if index[arm] == max_index]

        return random.choice(bestArms)

    def compute_index(self, arm_id):
        """the p-th quantile of the beta distribution for the arm
        """
        p = 1 - 1. / (self._n + 1)
        return special.btdtri(self.posterior[arm_id].alpha, self.posterior[arm_id].beta, p)


    def update(self, arm_id, *reward_args, **reward_kwargs):
        """Rewrite update function
        """
        self._rewards[arm_id].update(*reward_args, **reward_kwargs)
        self._counts[arm_id] += 1
        self._n += 1

        self.posterior[arm_id].update(*reward_args)  # update posterior distributions
        return self
