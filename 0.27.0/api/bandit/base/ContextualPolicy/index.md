# ContextualPolicy

Contextual bandit policy base class.

## Parameters

- **reward\_obj**

  *Type* → `RewardObj | None`

  *Default* → `None`

  The reward object used to measure the performance of each arm. This can be a metric, a statistic, or a distribution.
- **reward\_scaler**

  *Type* → `compose.TargetTransformRegressor | None`

  *Default* → `None`

  A reward scaler used to scale the rewards before they are fed to the reward object. This can be useful to scale the rewards to a (0, 1) range for instance.
- **burn\_in**

  *Default* → `0`

  The number of steps to use for the burn-in phase. Each arm is given the chance to be pulled during the burn-in phase. This is useful to mitigate selection bias.

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
> Update an arm's state.
>
> **Parameters**
>
> - **arm\_id**
> - **context**
> - **reward\_args**
> - **reward\_kwargs**
