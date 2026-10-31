# ChiSquared

Streaming Chi-squared statistic.

Maintains a contingency table between two variables `x` and `y` and computes the Chi-squared statistic incrementally. This can be used to measure the dependency between two categorical variables in a streaming setting.

## Attributes

- **degrees\_of\_freedom**

  Return the degrees of freedom of the contingency table.
- **name**
- **p\_value**

  Return the p-value associated with the Chi-squared statistic.

## Examples

```python
from river import stats

chi2 = stats.ChiSquared()

data = [
    ("A", 0),
    ("A", 0),
    ("B", 1),
    ("B", 1),
]

for x, y in data:
    chi2.update(x, y)

round(chi2.get(), 3)
```

```text
4.0
```

A rolling version can be obtained by wrapping with [`utils.Rolling`](<https://riverml.xyz/dev/api/utils/Rolling>):

```python
from river import utils

data = [
    ("A", 0),
    ("A", 0),
    ("B", 1),
    ("B", 1),
    ("C", 0),
]

chi2 = utils.Rolling(stats.ChiSquared, window_size=4)

for x, y in data:
    chi2.update(x, y)

round(chi2.get(), 3)
```

```text
4.0
```

## Methods

> [!NOTE]
>
> **get**
>
> Return the current value of the statistic.

> [!NOTE]
>
> **revert**

> [!NOTE]
>
> **update**
>
> Update the called instance.
>
> **Parameters**
>
> - **x** — `Hashable`
> - **y** — `Hashable`
