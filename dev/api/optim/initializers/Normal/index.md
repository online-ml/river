# Normal

Random normal initializer which simulate a normal distribution with specified parameters.

## Parameters

- **mu**

  *Type* → `float`

  *Default* → `0.0`

  The mean of the normal distribution
- **sigma**

  *Type* → `float`

  *Default* → `1.0`

  The standard deviation of the normal distribution
- **seed**

  *Type* → `int | None`

  *Default* → `None`

  Random number generation seed that can be set for reproducibility.

## Examples

```python
from river import optim

init = optim.initializers.Normal(mu=0, sigma=1, seed=42)

init(shape=1)
```

```text
np.float64(0.4967141...)
```

```python
init(shape=2)
```

```text
array([-0.1382643 ,  0.64768854])
```

## Methods

> [!NOTE]
>
> ****call****
>
> Returns a fresh set of weights.
>
> The return type is shape-dependent (a scalar when `shape == 1`, otherwise an array), and these values are routinely used as `collections.defaultdict` factories elsewhere, so the return is intentionally left dynamic.
>
> **Parameters**
>
> - **shape** — `int` — defaults to `1`
