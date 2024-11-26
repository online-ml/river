import math
import river.bandit

class KLUCB(river.bandit):

    def __init__(self, n_arms, horizon, c=0):
        self.n_arms = n_arms
        self.horizon = horizon
        self.c = c
        self.arm_count = [0 for _ in range (n_arms)]
        self.rewards = [0.0 for _ in range (n_arms)]
        self.arm = 0

    def update(self, arm, reward):
        self.arm_count[arm] += 1
        self.rewards[arm] += reward
        self.arm = arm

    def kl_divergence(self, p, q):
        if p == 0 :
            return float('inf') if q >=1 else -math.log(1-q)
        elif p == 1 :
            return float('inf') if q <=0 else -math.log(q)
        elif q<=0 or q>=1 :
            return float('inf')
        return p*math.log(p/q) + (1-p)*math.log((1-p)/(1-q))

    def kl_index(self, arm):
        self.arm_count[arm] = 1
        self.rewards[arm] = float('inf')