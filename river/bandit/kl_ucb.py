from __future__ import annotations

import math
import random


class KLUCB:
    """

    KL-UCB is an algorithm for solving the multi-armed bandit problem. It uses Kullback-Leibler (KL)
    divergence to calculate upper confidence bounds (UCBs) for each arm. The algorithm aims to balance
    exploration (trying different arms) and exploitation (selecting the best-performing arm) in a principled way.

    Parameters
    ----------
    n_arms (int):
        The total number of arms available for selection.
    horizon (int):
        The total number of time steps or trials during which the algorithm will run.
    c (float, default=0):
        A scaling parameter for the confidence bound. Larger values promote exploration,
        while smaller values favor exploitation.

    Attributes
    ----------
    arm_count (list[int]):
        A list where each element tracks the number of times an arm has been selected.
    rewards (list[float]):
        A list where each element accumulates the total rewards received from pulling each arm.
    t (int):
        The current time step in the algorithm.

    Methods
    -------
    update(arm, reward):
        Updates the statistics for the selected arm based on the observed reward.

    kl_divergence(p, q):
        Computes the Kullback-Leibler (KL) divergence between probabilities `p` and `q`.
        This measures how one probability distribution differs from another.

    kl_index(arm):
        Calculates the KL-UCB index for a specific arm using binary search to determine the upper bound.

    pull_arm(arm):
        Simulates pulling an arm by generating a reward based on the empirical mean reward for that arm.


    Examples:
    ----------

    >>> from river.bandit import KLUCB
    >>> n_arms = 3
    >>> horizon = 100
    >>> c = 1
    >>> klucb = KLUCB(n_arms=n_arms, horizon=horizon, c=c)

    >>> random.seed(42)

    >>> def calculate_reward(arm):
    ...         #Example: Bernoulli reward based on the true probability (for testing)
    ...         true_probabilities = [0.3, 0.5, 0.7]  # Example probabilities for each arm
    ...         return 1 if random.random() < true_probabilities[arm] else 0
    >>> # Initialize tracking variables
    >>> selected_arms = []
    >>> total_reward = 0
    >>> cumulative_rewards = []
    >>> for t in range(1, horizon + 1):
    ...     klucb.t = t
    ...     indices = [klucb.kl_index(arm) for arm in range(n_arms)]
    ...     chosen_arm = indices.index(max(indices))
    ...     reward = calculate_reward(chosen_arm)
    ...     klucb.update(chosen_arm, reward)
    ...     selected_arms.append(chosen_arm)
    ...     total_reward += reward
    ...     cumulative_rewards.append(total_reward)


    >>> print("Selected arms:", selected_arms)
    Selected arms: [0, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]



    >>> print("Cumulative rewards:", cumulative_rewards)
    Cumulative rewards: [0, 1, 2, 3, 3, 3, 3, 4, 5, 6, 7, 7, 8, 9, 9, 9, 10, 10, 10, 11, 11, 11, 11, 12, 12, 13, 14, 15, 15, 16, 16, 16, 17, 17, 18, 19, 19, 20, 20, 20, 20, 21, 22, 23, 24, 25, 26, 27, 27, 28, 29, 30, 31, 31, 31, 31, 32, 32, 33, 34, 34, 34, 34, 35, 35, 35, 36, 37, 38, 39, 40, 40, 40, 41, 41, 42, 42, 42, 43, 44, 44, 45, 45, 45, 46, 47, 47, 48, 49, 50, 51, 52, 52, 53, 54, 55, 55, 56, 56, 56]



    >>> print(f"Total Reward: {total_reward}")
    Total Reward: 56

    """

    def __init__(self, n_arms, horizon, c=0):
        self.n_arms = n_arms
        self.horizon = horizon
        self.c = c
        self.arm_count = [1 for _ in range(n_arms)]
        self.rewards = [0.0 for _ in range(n_arms)]
        self.t = 0

    def update(self, arm, reward):
        """
        Updates the number of times the arm has been pulled and the cumulative reward
        for the given arm. Also increments the current time step.

        Parameters
        ----------
        arm (int): The index of the arm that was pulled.
        reward (float): The reward obtained from pulling the arm.
        """
        self.arm_count[arm] += 1
        self.rewards[arm] += reward
        self.t += 1

    def kl_divergence(self, p, q):
        """
        Computes the Kullback-Leibler (KL) divergence between two probabilities `p` and `q`.

        Parameters
        ----------
        p (float): The first probability (true distribution).
        q (float): The second probability (approximated distribution).

        Returns
        -------
        float: The KL divergence value. Returns infinity if `q` is not a valid probability.
        """

        if p == 0:
            return float("inf") if q >= 1 else -math.log(1 - q)
        elif p == 1:
            return float("inf") if q <= 0 else -math.log(q)
        elif q <= 0 or q >= 1:
            return float("inf")
        return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))

    def kl_index(self, arm):
        """
        Computes the KL-UCB index for a given arm using binary search.
        This determines the upper confidence bound for the arm.

        Parameters
        ----------
        arm (int): The index of the arm to compute the index for.

        Returns
        -------
        float: The KL-UCB index for the arm.
        """

        n_t = self.arm_count[arm]
        if n_t == 0:
            return float("inf")  # Unseen arm
        empirical_mean = self.rewards[arm] / n_t
        log_t_over_n = math.log(self.t + 1) / n_t
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
        return low

    def pull_arm(self, arm):
        """
        Simulates pulling an arm by generating a reward based on its empirical mean.

        Parameters
        ----------
        arm (int): The index of the arm to pull.

        Returns
        -------
        int: 1 if the arm yields a reward, 0 otherwise.
        """
        prob = self.rewards[arm] / self.arm_count[arm]
        return 1 if random.random() < prob else 0

    @staticmethod
    def _unit_test_params():
        """
        Returns a list of dictionaries with parameters to initialize the KLUCB class
        for unit testing.
        """
        return [
            {"n_arms": 2, "horizon": 100, "c": 0.5},
            {"n_arms": 5, "horizon": 1000, "c": 1.0},
            {"n_arms": 10, "horizon": 500, "c": 0.1},
        ]
