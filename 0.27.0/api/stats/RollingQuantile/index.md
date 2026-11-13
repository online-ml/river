# RollingQuantile

Running quantile over a window.

## Parameters

- **q**

  *Type* → `float`

  Determines which quantile to compute, must be comprised between 0 and 1.
- **window\_size**

  *Type* → `int`

  Size of the window.

## Attributes

- **name**
- **window\_size**

  The size of the rolling window.

## Examples

```python
from river import stats

rolling_quantile = stats.RollingQuantile(
    q=.5,
    window_size=101,
)

for i in range(1001):
    rolling_quantile.update(i)
    if i % 100 == 0:
        print(rolling_quantile.get())
```

```text
0.0
50.0
150.0
250.0
350.0
450.0
550.0
650.0
750.0
850.0
950.0
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

## References

- [Left sorted](<https://stackoverflow.com/questions/8024571/insert-an-item-into-sorted-list-in-python>)
