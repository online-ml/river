# evaluate

Benchmark a list of policies on a given Gym environment.

This is a high-level utility function for benchmarking a list of policies on a given Gym environment. For example, it can be used to populate a `pandas.DataFrame` with the contents of each step of each episode.

## Parameters

- **policies**

  *Type* → `list[bandit.base.Policy]`

  A list of policies to evaluate. The policy will be reset before each episode.
- **env**

  *Type* → `gym.Env`

  The Gym environment to use. One copy will be made for each policy at the beginning of each episode.
- **reward\_stat**

  *Type* → `stats.base.Univariate | None`

  *Default* → `None`

  A univariate statistic to keep track of the rewards. This statistic will be reset before each episode. Note that this is not the same as the reward object used by the policies. It's just a statistic to keep track of each policy's performance. If `None`, `stats.Sum` is used.
- **n\_episodes**

  *Type* → `int`

  *Default* → `20`

  The number of episodes to run.
- **seed**

  *Type* → `int | None`

  *Default* → `None`

  Random number generator seed for reproducibility. A random number generator will be used to seed differently the environment before each episode.

## Examples

```python
import gymnasium as gym
from river import bandit

trace = bandit.evaluate(
    policies=[
        bandit.UCB(delta=1, seed=42),
        bandit.EpsilonGreedy(epsilon=0.1, seed=42),
    ],
    env=gym.make(
        'river_bandits/CandyCaneContest-v0',
        max_episode_steps=100
    ),
    n_episodes=5,
    seed=42
)

for step in trace:
    print(step)
    break
```

```text
{'episode': 0, 'step': 0, 'policy_idx': 0, 'arm': 81, 'reward': 0.0, 'reward_stat': 0.0}
```

The return type of this function is a generator. Each step of the generator is a dictionary. You can pass the generator to a `pandas.DataFrame` to get a nice representation of the results.

```python
import pandas as pd

trace = bandit.evaluate(
    policies=[
        bandit.UCB(delta=1, seed=42),
        bandit.EpsilonGreedy(epsilon=0.1, seed=42),
    ],
    env=gym.make(
        'river_bandits/CandyCaneContest-v0',
        max_episode_steps=100
    ),
    n_episodes=5,
    seed=42
)

trace_df = pd.DataFrame(trace)
trace_df.sample(5, random_state=42)
```

```text
     episode  step  policy_idx  arm  reward  reward_stat
521        2    60           1   25     0.0         36.0
737        3    68           1   40     1.0         20.0
740        3    70           0   58     0.0         36.0
660        3    30           0   31     1.0         16.0
411        2     5           1   35     1.0          5.0
```

The length of the dataframe is the number of policies times the number of episodes times the maximum number of steps per episode.

```python
len(trace_df)
```

```text
1000
```

```python
(
    trace_df.policy_idx.nunique() *
    trace_df.episode.nunique() *
    trace_df.step.nunique()
)
```

```text
1000
```
