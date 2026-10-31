# RollingIQR

Computes the rolling interquartile range.

## Parameters

- **window\_size**

  *Type* → `int`

  Size of the window.
- **q\_inf**

  *Type* → `float`

  *Default* → `0.25`

  Desired inferior quantile, must be between 0 and 1. Defaults to `0.25`.
- **q\_sup**

  *Type* → `float`

  *Default* → `0.75`

  Desired superior quantile, must be between 0 and 1. Defaults to `0.75`.

## Attributes

- **name**
- **window\_size**

  The size of the rolling window.

## Examples

```python
from river import stats
rolling_iqr = stats.RollingIQR(
    q_inf=0.25,
    q_sup=0.75,
    window_size=101
)

for i in range(0, 1001):
    rolling_iqr.update(i)
    if i % 100 == 0:
        print(rolling_iqr.get())
```

```text
0.0
50.0
50.0
50.0
50.0
50.0
50.0
50.0
50.0
50.0
50.0
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
