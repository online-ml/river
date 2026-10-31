# BayesUCB

Bayes-UCB bandit policy.

Bayes-UCB is a Bayesian algorithm for the multi-armed bandit problem. It uses the posterior distribution of the reward of each arm to compute an upper confidence bound (UCB) on the expected reward of each arm. The arm with the highest UCB is then pulled. The posterior distribution is updated after each pull. The algorithm is described in [^/dev/api/bandit/BayesUCB/#1].

## Parameters

- **reward\_obj**

  *Default* → `None`

  The reward object that is used to update the posterior distribution.
- **burn\_in**

  *Default* → `0`

  Number of initial observations per arm before using the posterior distribution.
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

policy = bandit.BayesUCB(seed=123)

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
Sum: 841.
```

## Methods

> [!NOTE]
>
> **compute\_index**
>
> the p-th quantile of the beta distribution for the arm
>
> **Parameters**
>
> - **arm\_id**

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
> Rewrite update function
>
> **Parameters**
>
> - **arm\_id**
> - **reward\_args**
> - **reward\_kwargs**

[^/dev/api/bandit/BayesUCB/#1]: [Kaufmann, Emilie, Olivier Cappé, and Aurélien Garivier. "On Bayesian upper confidence bounds for bandit problems." Artificial intelligence and statistics. PMLR, 2012.](<http://proceedings.mlr.press/v22/kaufmann12/kaufmann12.pdf>)
