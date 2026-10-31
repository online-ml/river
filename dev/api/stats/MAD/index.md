# MAD

Median Absolute Deviation (MAD).

The median absolute deviation is the median of the absolute differences between each data point and the data's overall median. In an online setting, the median of the data is unknown beforehand. Therefore, both the median of the data and the median of the differences of the data with respect to the latter are updated online. To be precise, the median of the data is updated before the median of the differences. As a consequence, this online version of the MAD does not coincide exactly with its batch counterpart.

## Attributes

- **median** (`stats.Median`)

  The median of the data.

## Examples

```python
from river import stats

X = [4, 2, 5, 3, 0, 4]

mad = stats.MAD()
for x in X:
    mad.update(x)
    print(mad.get())
```

```text
0.0
2.0
1.0
1.0
1.0
1.0
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

- [Median absolute deviation article on Wikipedia](<https://en.wikipedia.org/wiki/Median_absolute_deviation>)
