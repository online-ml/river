import math
import river.bandit

class Klucb(river.bandit):

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
        n_t = self.counts[arm]
        if n_t == 0:
            return float('inf')  # Unseen arm
        empirical_mean = self.rewards[arm] / n_t
        log_t_over_n = math.log(self.t) / n_t
        c_factor = self.c * log_t_over_n

        # Binary search to find the q that satisfies the KL-UCB condition
        low = empirical_mean
        high = 1.0
        for _ in range(100):  # Fixed number of iterations for binary search
            mid = (low + high) / 2
            kl = self._kl_divergence(empirical_mean, mid)
            if kl > c_factor:
                high = mid
            else:
                low = mid
        return low
