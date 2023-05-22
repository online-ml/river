import random
from random import betavariate
from river import bandit
from scipy.special import btdtri


class Beta:
    """
    Posterior of Bernoulli/Beta distribution
    """
    def __init__(self, a=1, b=1):
        self.a = a
        self.b = b

    def reset(self, a=0, b=0):
        if a == 0:
            a = self.a
        if b == 0:
            b = self.b
        self.N = [a, b]

    def update(self, obs):
        self.N[int(obs)] += 1

    def sample(self):
        return betavariate(self.N[1], self.N[0])

    def quantile(self, p):
        return btdtri(self.N[1], self.N[0], p)


class BayesUCB(bandit.base.Policy):
    """
    Bayes-UCB strategy for bandit algorithm proposed by Kaufmann E. et al.

    Parameters
    --------
    n_arms : int. Number of arms.
    t : int. Time.
    posterior : dict. Contains the posterior distributions of all the arms.
    reward_obj : The reward object used to measure the performance of each arm. This can be a metric, a statistic, or a distribution.
    burn_in : The number of steps to use for the burn-in phase. Each arm is given the chance to be pulled during the burn-in phase. This is useful to mitigate selection bias.

    Examples
    --------
    env = gym.make('river_bandits/CandyCaneContest-v0')
    _ = env.reset(seed=42)
    _ = env.action_space.seed(123)

    policy = BayesUCB(n_arms=env.action_space.n)

    metric = stats.Sum()
    while True:
        action = next(policy.pull(range(env.action_space.n)))
        observation, reward, terminated, truncated, info = env.step(action)
        policy = policy.update(action, reward)
        policy.getReward(action, reward)
        metric = metric.update(reward)
        if terminated or truncated:
            break

    print('Sum:', metric.get())
    """
    def __init__(self, n_arms: int, reward_obj=None, burn_in=0):
        super().__init__(reward_obj, burn_in)
        self.n_arms = n_arms
        self.t = 1
        self.posterior = dict()
        for arm_id in range(self.n_arms):
            self.posterior[arm_id] = Beta()
            self.posterior[arm_id].reset()

    def _pull(self, arm_ids):
        index = dict()
        for arm_id in arm_ids:
            index[arm_id] = self.computeIndex(arm_id)
        maxIndex = max(index.values())
        bestArms = [arm for arm in index.keys() if index[arm] == maxIndex]
        return random.choice(bestArms)

    def computeIndex(self, arm_id):
        return self.posterior[arm_id].quantile(1 - 1. / self.t)

    def getReward(self, arm, reward):
        self.posterior[arm].update(reward)
        self.t += 1
