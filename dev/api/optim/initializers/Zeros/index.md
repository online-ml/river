# Zeros

Constant initializer which always returns zeros.

## Examples

```python
from river import optim

init = optim.initializers.Zeros()

init(shape=1)
```

```text
0.0
```

```python
init(shape=2)
```

```text
array([0., 0.])
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
