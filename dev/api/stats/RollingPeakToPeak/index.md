# RollingPeakToPeak

Running peak to peak (max - min) over a window.

## Parameters

- **window\_size**

  *Type* → `int`

  Size of the rolling window.

## Attributes

- **max** ([`stats.RollingMax`](<https://riverml.xyz/dev/api/stats/RollingMax>))

  The running rolling max.
- **min** ([`stats.RollingMin`](<https://riverml.xyz/dev/api/stats/RollingMin>))

  The running rolling min.

## Examples

```python
from river import stats

X = [1, -4, 3, -2, 2, 1]
ptp = stats.RollingPeakToPeak(window_size=2)
for x in X:
    ptp.update(x)
    print(ptp.get())
```

```text
0
5
7
5
4
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
