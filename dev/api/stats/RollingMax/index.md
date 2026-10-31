# RollingMax

Running max over a window.

## Parameters

- **window\_size**

  *Type* → `int`

  Size of the rolling window.

## Attributes

- **name**
- **window\_size**

  The size of the rolling window.

## Examples

```python
from river import stats

X = [1, -4, 3, -2, 2, 1]
rolling_max = stats.RollingMax(window_size=2)
for x in X:
    rolling_max.update(x)
    print(rolling_max.get())
```

```text
1
1
3
3
2
2
```

## Methods

> [!NOTE]
>
> **get**
>
> Return the current value of the statistic.

> [!NOTE]
>
> **update**
>
> Update the called instance.
>
> **Parameters**
>
> - **x** — `float`
