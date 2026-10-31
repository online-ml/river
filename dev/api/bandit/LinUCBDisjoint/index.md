# LinUCBDisjoint

LinUCB, disjoint variant.

Although it works, as of yet it is too slow to realistically be used in practice.

The way this works is that each arm is assigned a [`linear_model.BayesianLinearRegression`](<https://riverml.xyz/dev/api/linear-model/BayesianLinearRegression>) instance. This instance is updated every time the arm is pulled. The context is used as features for the regression. The reward is used as the target. The posterior distribution is used to compute the upper confidence bound. The arm with the highest upper confidence bound is pulled.

## Parameters

- **alpha**

  *Type* → `float`

  *Default* → `1.0`

  Parameter used in each Bayesian linear regression.
- **beta**

  *Type* → `float`

  *Default* → `1.0`

  Parameter used in each Bayesian linear regression.
- **smoothing**

  *Type* → `float | None`

  *Default* → `None`

  Parameter used in each Bayesian linear regression.
- **reward\_obj**

  *Default* → `None`

  The reward object used to measure the performance of each arm.
- **burn\_in**

  *Default* → `0`

  The number of time steps during which each arm is pulled once.
- **seed**

  *Type* → `int | None`

  *Default* → `None`

  Random number generator seed for reproducibility.

## Attributes

- **ranking**

  Return the list of arms in descending order of performance.

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
> - **context** — `dict | None` — defaults to `None`
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
> - **context**
> - **reward\_args**
> - **reward\_kwargs**

## References

- [A Contextual-Bandit Approach to Personalized News Article Recommendation](<https://arxiv.org/abs/1003.0146>)
- [Contextual Bandits Analysis of LinUCB Disjoint Algorithm with Dataset](<https://kfoofw.github.io/contextual-bandits-linear-ucb-disjoint/>)
