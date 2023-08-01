from __future__ import annotations

import copy
import random
import typing

try:
    import gym
except ImportError:
    ...

from river import bandit, stats

from .base import ArmID


def evaluate(
    policies: list[bandit.base.Policy],
    env: gym.Env,
    reward_stat: stats.base.Univariate | None = None,
    n_episodes: int = 20,
    seed: int | None = None,
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

    >>> trace = bandit.evaluate(
    ...     policies=[
    ...         bandit.UCB(delta=1, seed=42),
    ...         bandit.EpsilonGreedy(epsilon=0.1, seed=42),
    ...     ],
    ...     env=gym.make(
    ...         'river_bandits/CandyCaneContest-v0',
    ...         max_episode_steps=100
    ...     ),
    ...     n_episodes=5,
    ...     seed=42
    ... )

    >>> for step in trace:
    ...     print(step)
    ...     break
    {'episode': 0, 'step': 0, 'policy_idx': 0, 'arm': 81, 'reward': 0.0, 'reward_stat': 0.0}

    The return type of this function is a generator. Each step of the generator is a dictionary.
    You can pass the generator to a `pandas.DataFrame` to get a nice representation of the results.

    >>> import pandas as pd

    >>> trace = bandit.evaluate(
    ...     policies=[
    ...         bandit.UCB(delta=1, seed=42),
    ...         bandit.EpsilonGreedy(epsilon=0.1, seed=42),
    ...     ],
    ...     env=gym.make(
    ...         'river_bandits/CandyCaneContest-v0',
    ...         max_episode_steps=100
    ...     ),
    ...     n_episodes=5,
    ...     seed=42
    ... )

    >>> trace_df = pd.DataFrame(trace)
    >>> trace_df.sample(5, random_state=42)
         episode  step  policy_idx  arm  reward  reward_stat
    521        2    60           1   25     0.0         36.0
    737        3    68           1   40     1.0         20.0
    740        3    70           0   58     0.0         36.0
    660        3    30           0   31     1.0         16.0
    411        2     5           1   35     1.0          5.0

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
            for policy_idx, (policy_, env_, reward_stat_) in enumerate(
                zip(episode_policies, episode_envs, episode_reward_stats)
            ):
                if done[policy_idx]:
                    continue

                arm = policy_.pull(range(env_.action_space.n))  # type: ignore[attr-defined]
                observation, reward, terminated, truncated, info = env_.step(arm)
                policy_.update(arm, reward)
                reward_stat_.update(reward)

                yield {
                    "episode": episode,
                    "step": step,
                    "policy_idx": policy_idx,
                    "arm": arm,
                    "reward": reward,
                    "reward_stat": reward_stat_.get(),
                }

                done[policy_idx] = terminated or truncated
            step += 1


History = typing.Iterator[
    tuple[
        list[ArmID],  # arms available to pull from
        dict,  # context
        ArmID,  # arm that was pulled
        float,  # reward
    ]
]


def evaluate_offline(
    policy: bandit.base.Policy,
    history: History | bandit.datasets.BanditDataset,
    reward_stat: stats.base.Univariate = None,
) -> tuple[stats.base.Univariate, int]:
    """Evaluate a policy on historical logs using replay.

    This is a high-level utility function for evaluating a policy using the replay methodology.
    This methodology is an off-policy evaluation method. It does not require an environment, and is
    instead data-driven.

    At each step, an arm is pulled from the provided policy. If the arm is the same as the arm
    that was pulled in the historical data, the reward is used to update the policy. If the arm is
    different, the reward is ignored. This is the off-policy aspect of the evaluation.

    Parameters
    ----------
    policy
        The policy to evaluate.
    history
        The history of the bandit problem. This is a generator that yields tuples of the form
        `(arms, context, arm, reward)`.
    reward_stat
        The reward statistic to use. Defaults to `stats.Sum`.

    Returns
    -------
    The reward statistic, along with the number of times the policy pulled the same arm as the
    historical data.

    Examples
    --------

    >>> import random
    >>> from river import bandit

    >>> rng = random.Random(42)
    >>> arms = ['A', 'B', 'C']
    >>> clicks = [
    ...     (
    ...         arms,
    ...         # no context
    ...         None,
    ...         # random arm
    ...         rng.choice(arms),
    ...         # reward
    ...         rng.random() > 0.5
    ...     )
    ...     for _ in range(1000)
    ... ]

    >>> total_reward, n_samples_used = bandit.evaluate_offline(
    ...     policy=bandit.EpsilonGreedy(0.1, seed=42),
    ...     history=clicks,
    ... )

    >>> total_reward
    Sum: 172.

    >>> n_samples_used
    321

    This also works out of the box with datasets that inherit from `river.bandit.BanditDataset`.

    >>> news = bandit.datasets.NewsArticles()
    >>> total_reward, n_samples_used = bandit.evaluate_offline(
    ...     policy=bandit.RandomPolicy(seed=42),
    ...     history=news,
    ... )

    >>> total_reward, n_samples_used
    (Sum: 105., 1027)

    As expected, the policy succeeds in roughly 10% of cases. Indeed, there are 10 arms and 10000
    samples, so the expected number of successes is 1000.

    References
    ----------
    [^1]: [Offline Evaluation of Multi-Armed Bandit Algorithms in Python using Replay](https://jamesrledoux.com/algorithms/offline-bandit-evaluation/)
    [^2]: [Unbiased Offline Evaluation of Contextual-bandit-based News Article Recommendation Algorithms](https://arxiv.org/pdf/1003.5956.pdf)
    [^3]: [Understanding Inverse Propensity Score for Contextual Bandits](https://chanind.github.io/ai/2022/03/14/inverse-propensity-score-contextual-bandits.html)

    """

    if isinstance(history, bandit.datasets.BanditDataset):
        arms = history.arms
        history = ((arms, context, arm, reward) for context, arm, reward in history)

    reward_stat = reward_stat or stats.Sum()

    is_contextual = isinstance(policy, bandit.base.ContextualPolicy)

    for arms_available, context, chosen_arm, reward in history:
        probability = 1  # TODO: use inverse propensity scoring
        arm = (
            policy.pull(arms_available, context=context)  # type: ignore[call-arg]
            if is_contextual
            else policy.pull(arms_available)
        )

        # Do nothing if the chosen arm differs from the arm in the historical data
        if chosen_arm != arm:
            continue

        if is_contextual:
            policy.update(arm, context, reward)
        else:
            policy.update(arm, reward)
        reward_stat.update(reward / probability)  # type: ignore

    # Normalize the reward statistic by the number of times the policy pulled the same arm as what
    # was pulled in the historical data
    return reward_stat, policy._n
