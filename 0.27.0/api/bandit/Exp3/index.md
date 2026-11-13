# Exp3

Exp3 bandit policy.

This policy works by maintaining a weight for each arm. These weights are used to randomly decide which arm to pull. The weights are increased or decreased, depending on the reward. An egalitarianism factor \\(\\gamma \\in \\\[0, 1\\\]\\) is included, to tune the desire to pick an arm uniformly at random. That is, if \\(\\gamma = 1\\), the arms are picked uniformly at random.

## Parameters

- **gamma**

  *Type* → `float`

  The egalitarianism factor. Setting this to 0 leads to what is called the EXP3 policy.
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
from river import proba
from river import stats

env = gym.make(
    'river_bandits/CandyCaneContest-v0'
)
_ = env.reset(seed=42)
_ = env.action_space.seed(123)

policy = bandit.Exp3(gamma=0.5, seed=42)

metric = stats.Sum()
while True:
    action = policy.pull(range(env.action_space.n))
    observation, reward, terminated, truncated, info = env.step(action)
    policy.update(action, reward)
    metric.update(reward)
    if terminated or truncated:
        break

metric
```

```text
Sum: 799.
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

- [Auer, P., Cesa-Bianchi, N., Freund, Y. and Schapire, R.E., 2002. The nonstochastic multiarmed bandit problem. SIAM journal on computing, 32(1), pp.48-77.](<https://cseweb.ucsd.edu/~yfreund/papers/bandits.pdf>)
- [Adversarial Bandits and the Exp3 Algorithm — Jeremy Kun](<https://jeremykun.com/2013/11/08/adversarial-bandits-and-the-exp3-algorithm/>)
