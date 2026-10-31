# Constant

Constant initializer which always returns the same value.

## Parameters

- **value**

  *Type* → `float`

## Examples

```python
from river import optim

init = optim.initializers.Constant(value=3.14)

init(shape=1)
```

```text
3.14
```

```python
init(shape=2)
```

```text
array([3.14, 3.14])
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
