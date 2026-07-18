import math

from marks import benchmark

from river import bandit


@benchmark("bandit")
def test_epsilon_greedy_pull_update(benchmark) -> None:
    arms = [0, 1, 2]
    rewards = {0: 0.1, 1: 0.4, 2: 0.2}

    def run() -> None:
        policy = bandit.EpsilonGreedy(epsilon=0.1, seed=42)
        for t in range(2_000):
            arm = policy.pull(arms)
            reward = rewards[arm] + math.sin(t)
            policy.update(arm, reward)

    benchmark(run)
