# PeakToPeak

Running peak to peak (max - min).

## Attributes

- **name**

## Examples

```python
from river import stats

X = [1, -4, 3, -2, 2, 4]
ptp = stats.PeakToPeak()
for x in X:
    ptp.update(x)
    print(ptp.get())
```

```text
0.
5.
7.
7.
7.
8.
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
