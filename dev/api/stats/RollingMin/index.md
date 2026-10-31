# RollingMin

Running min over a window.

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
rolling_min = stats.RollingMin(2)
for x in X:
    rolling_min.update(x)
    print(rolling_min.get())
```

```text
1
-4
-4
-2
-2
1
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
