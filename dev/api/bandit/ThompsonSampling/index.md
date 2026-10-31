# ThompsonSampling

Thompson sampling.

Thompson sampling is often used with a Beta distribution. However, any probability distribution can be used, as long it makes sense with the reward shape. For instance, a Beta distribution is meant to be used with binary rewards, while a Gaussian distribution is meant to be used with continuous rewards.

The randomness of a distribution is controlled by its seed. The seed should not set within the distribution, but should rather be defined in the policy parametrization. In other words, you should do this:

```python
policy = ThompsonSampling(dist=proba.Beta(1, 1), seed=42) 
```

and not this:

```python
policy = ThompsonSampling(dist=proba.Beta(1, 1, seed=42)) 
```

## Parameters

- **reward\_obj**

  *Type* → `proba.base.Distribution | None`

  *Default* → `None`

  A distribution to sample from.
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
from river import proba
from river import stats

env = gym.make(
    'river_bandits/CandyCaneContest-v0'
)
_ = env.reset(seed=42)
_ = env.action_space.seed(123)

policy = bandit.ThompsonSampling(reward_obj=proba.Beta(), seed=101)

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
Sum: 820.
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

- [An Empirical Evaluation of Thompson Sampling](<https://proceedings.neurips.cc/paper/2011/file/e53a0a2978c28872a4505bdb51db06dc-Paper.pdf>)
