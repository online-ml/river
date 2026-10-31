# EpsilonGreedy

\\(\\varepsilon\\)-greedy bandit policy.

Performs arm selection by using an \\(\\varepsilon\\)-greedy bandit strategy. An arm is selected at each step. The best arm is selected (1 - \\(\\varepsilon\\))% of the time.

Selection bias is a common problem when using bandits. This bias can be mitigated by using burn-in phase. Each model is given the chance to learn during the first `burn_in` steps.

## Parameters

- **epsilon**

  *Type* → `float`

  The probability of exploring.
- **decay**

  *Default* → `0.0`

  The decay rate of epsilon.
- **reward\_obj**

  *Default* → `None`

  The reward object used to measure the performance of each arm. This can be a metric, a statistic, or a distribution.
- **burn\_in**

  *Default* → `0`

  The number of steps to use for the burn-in phase. Each arm is given the chance to be pulled during the burn-in phase. This is useful to mitigate selection bias.
- **seed**

  *Type* → `int | None`

  *Default* → `None`

  Random number generator seed for reproducibility.

## Attributes

- **current\_epsilon**

  The value of epsilon after factoring in the decay rate.
- **ranking**

  Return the list of arms in descending order of performance.

## Examples

```python
import gymnasium as gym
from river import bandit
from river import stats

env = gym.make(
    'river_bandits/CandyCaneContest-v0'
)
_ = env.reset(seed=42)
_ = env.action_space.seed(123)

policy = bandit.EpsilonGreedy(epsilon=0.9, seed=101)

metric = stats.Sum()
while True:
    arm = policy.pull(range(env.action_space.n))
    observation, reward, terminated, truncated, info = env.step(arm)
    policy.update(arm, reward)
    metric.update(reward)
    if terminated or truncated:
        break

metric
```

```text
Sum: 775.
```

## Methods

> [!NOTE]
>
> **pull**
>
> Pull arm(s).
>
> This method is a generator that yields the arm(s) that should be pulled. During the burn-in phase, all the arms that have not been pulled enough times are yielded. Once the burn-in phase is over, the policy is allowed to choose the arm(s) that should be pulled. If you only want to pull one arm at a time during the burn-in phase, simply call `next(policy.pull(arms))`.
>
> **Parameters**
>
> - **arm\_ids** — `list[ArmID]`
>
> **Returns**
>
> `ArmID`: A single arm.

> [!NOTE]
>
> **update**
>
> Update an arm's state.
>
> **Parameters**
>
> - **arm\_id**
> - **reward\_args**
> - **reward\_kwargs**

## References

- [ε-Greedy Algorithm - The Multi-Armed Bandit Problem and Its Solutions - Lilian Weng](<https://lilianweng.github.io/lil-log/2018/01/23/the-multi-armed-bandit-problem-and-its-solutions.html#ε-greedy-algorithm>)
