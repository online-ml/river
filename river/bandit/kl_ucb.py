import math
import random
import river.bandit

class KLUCB():
    "define the main class for the Klucb"

    def __init__(self, n_arms, horizon, c):
        self.n_arms = n_arms
        self.horizon = horizon
        self.c = c
        self.arm_count = [1 for _ in range (n_arms)]
        self.rewards = [0.0 for _ in range (n_arms)]
        self.arm = 0

    def calculate_reward(self, arm):
        return 1 if random.random() < self.rewards[arm] else 0

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
        n_t = self.arm_count[arm]
        if n_t == 0:
            return float('inf')  # Unseen arm
        empirical_mean = self.rewards[arm] / n_t
        log_t_over_n = math.log(math.log(arm)) / n_t
        c_factor = self.c * log_t_over_n

        # Binary search to find the q that satisfies the KL-UCB condition
        low = empirical_mean
        high = 1.0
        for _ in range(100):  # Fixed number of iterations for binary search
            mid = (low + high) / 2
            kl = self.kl_divergence(empirical_mean, mid)
            if kl > c_factor:
                high = mid
            else:
                low = mid

        selected_arm = low
        return selected_arm
