# UCB

Upper Confidence Bound (UCB) bandit policy.

Due to the nature of this algorithm, it's recommended to scale the target so that it exhibits sub-gaussian properties. This can be done by passing a [`preprocessing.TargetStandardScaler`](<https://riverml.xyz/dev/api/preprocessing/TargetStandardScaler>) instance to the `reward_scaler` argument.

## Parameters

- **delta**

  *Type* → `float`

  The confidence level. Setting this to 1 leads to what is called the UCB1 policy.
- **reward\_obj**

  *Default* → `None`

  The reward object used to measure the performance of each arm. This can be a metric, a statistic, or a distribution.
- **reward\_scaler**

  *Default* → `None`

  A reward scaler used to scale the rewards before they are fed to the reward object. This can be useful to scale the rewards to a (0, 1) range for instance.
- **burn\_in**

  *Default* → `0`

  The number of steps to use for the burn-in phase. Each arm is given the chance to be pulled during the burn-in phase. This is useful to mitigate selection bias.
- **seed**

  *Type* → `int | None`

  *Default* → `None`

  Random number generator seed for reproducibility.

## Attributes

- **ranking**

  Return the list of arms in descending order of performance.

## Examples

```python
import gymnasium as gym
from river import bandit
from river import preprocessing
from river import stats

env = gym.make(
    'river_bandits/CandyCaneContest-v0'
)
_ = env.reset(seed=42)
_ = env.action_space.seed(123)

policy = bandit.UCB(
    delta=100,
    reward_scaler=preprocessing.TargetStandardScaler(None),
    seed=42
)

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
Sum: 744.
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

- [Lai, T. L., &amp; Robbins, H. (1985). Asymptotically efficient adaptive allocation rules. Advances in applied mathematics, 6(1), 4-22.](<https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.674.1620&amp;rep=rep1&amp;type=pdf>)
- [Upper Confidence Bounds - The Multi-Armed Bandit Problem and Its Solutions - Lilian Weng](<https://lilianweng.github.io/lil-log/2018/01/23/the-multi-armed-bandit-problem-and-its-solutions.html#upper-confidence-bounds>)
- [The Upper Confidence Bound Algorithm - Bandit Algorithms](<https://banditalgs.com/2016/09/18/the-upper-confidence-bound-algorithm/>)
