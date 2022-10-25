import copy
import random
from typing import Callable, List

try:
    import gym
except ImportError:
    ...
from river import bandit, stats


def evaluate(
    policies: List[bandit.base.Policy],
    env: "gym.Env",
    pull_func: Callable[[bandit.base.Policy, "gym.Env"], bandit.base.ArmID],
    reward_stat: stats.base.Univariate = None,
    n_episodes: int = 20,
    seed: int = None,
):
    """Benchmark a list of policies on a given Gym environment.

    This is a high-level utility function for benchmarking a list of policies on a given
    Gym environment. For example, it can be used to populate a `pandas.DataFrame` with the contents
    of each step of each episode.

    Parameters
    ----------
    policies
        A list of policies to evaluate. The policy will be reset before each episode.
    env
        The Gym environment to use. One copy will be made for each policy at the beginning of each
        episode.
    pull_func
        A function that takes a policy and an environment as arguments and returns the arm that was
        pulled. This function is called at each time step for each policy. This is required because
        there is no standard way to pull an arm in Gym environments.
    reward_stat
        A univariate statistic to keep track of the rewards. This statistic will be reset before
        each episode. Note that this is not the same as the reward object used by the policies.
        It's just a statistic to keep track of each policy's performance. If `None`, `stats.Sum`
        is used.
    n_episodes
        The number of episodes to run.
    seed
        Random number generator seed for reproducibility. A random number generator will be used
        to seed differently the environment before each episode.

    Examples
    --------

    >>> import gym
    >>> from river import bandit

    >>> def pull_func(policy, env):
    ...     return next(policy.pull(range(env.action_space.n)))

    >>> trace = bandit.evaluate(
    ...     policies=[
    ...         bandit.UCB(delta=1),
    ...         bandit.EpsilonGreedy(epsilon=0.1, seed=42),
    ...     ],
    ...     env=gym.make(
    ...         'river_bandits/CandyCaneContest-v0',
    ...         max_episode_steps=100
    ...     ),
    ...     pull_func=pull_func,
    ...     n_episodes=5,
    ...     seed=42
    ... )

    >>> for step in trace:
    ...     print(step)
    ...     break
    {'episode': 0, 'step': 0, 'policy_idx': 0, 'action': 0, 'reward': 0.0, 'reward_stat': 0.0}

    The return type of this function is a generator. Each step of the generator is a dictionary.
    You can pass the generator to a `pandas.DataFrame` to get a nice representation of the results.

    >>> import pandas as pd

    >>> trace = bandit.evaluate(
    ...     policies=[
    ...         bandit.UCB(delta=1),
    ...         bandit.EpsilonGreedy(epsilon=0.1, seed=42),
    ...     ],
    ...     env=gym.make(
    ...         'river_bandits/CandyCaneContest-v0',
    ...         max_episode_steps=100
    ...     ),
    ...     pull_func=pull_func,
    ...     n_episodes=5,
    ...     seed=42
    ... )

    >>> trace_df = pd.DataFrame(trace)
    >>> trace_df.sample(5, random_state=42)
         episode  step  policy_idx  action  reward  reward_stat
    521        2    60           1      25     0.0         36.0
    737        3    68           1      40     1.0         20.0
    740        3    70           0      70     1.0         33.0
    660        3    30           0      30     1.0         13.0
    411        2     5           1      35     1.0          5.0

    The length of the dataframe is the number of policies times the number of episodes times the
    maximum number of steps per episode.

    >>> len(trace_df)
    1000

    >>> (
    ...     trace_df.policy_idx.nunique() *
    ...     trace_df.episode.nunique() *
    ...     trace_df.step.nunique()
    ... )
    1000

    """

    reward_stat = reward_stat or stats.Sum()
    rng = random.Random(seed)

    for episode in range(n_episodes):
        episode_policies = [policy.clone() for policy in policies]
        episode_env = copy.deepcopy(env)
        episode_env.reset(seed=rng.randint(0, 2**32))
        episode_env.action_space.seed(rng.randint(0, 2**32 - 1))
        episode_envs = [copy.deepcopy(episode_env) for _ in episode_policies]
        episode_reward_stats = [reward_stat.clone() for _ in policies]

        step = 0
        done = [False] * len(policies)

        while not all(done):
            for policy_idx, (_policy, _env, _reward_stat) in enumerate(
                zip(episode_policies, episode_envs, episode_reward_stats)
            ):
                if done[policy_idx]:
                    continue

                action = pull_func(_policy, _env)
                observation, reward, terminated, truncated, info = _env.step(action)
                _policy.update(action, reward)
                _reward_stat.update(reward)

                yield {
                    "episode": episode,
                    "step": step,
                    "policy_idx": policy_idx,
                    "action": action,
                    "reward": reward,
                    "reward_stat": _reward_stat.get(),
                }

                done[policy_idx] = terminated or truncated
            step += 1
