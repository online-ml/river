# Kurtosis

Running kurtosis using Welford's algorithm.

## Parameters

- **bias**

  *Type* → `bool`

  *Default* → `False`

  If `False`, then the calculations are corrected for statistical bias.

## Attributes

- **name**

## Examples

```python
from river import stats
import scipy.stats
import numpy as np

np.random.seed(42)
X = np.random.normal(loc=0, scale=1, size=10)

kurtosis = stats.Kurtosis(bias=False)
for x in X:
    kurtosis.update(x)
    print(kurtosis.get())
```

```text
-3.0
-2.0
-1.5
1.4130027920707047
0.15367976585756438
0.46142633246812653
-1.620647789230658
-1.3540178492487054
-1.2310268787102745
-0.9490372374384453
```

```python
for i in range(2, len(X)+1):
    print(scipy.stats.kurtosis(X[:i], bias=False))
```

```text
-2.0
-1.4999999999999998
1.4130027920707082
0.15367976585756082
0.46142633246812403
-1.620647789230658
-1.3540178492487063
-1.2310268787102738
-0.9490372374384459
```

```python
kurtosis = stats.Kurtosis(bias=True)
for x in X:
    kurtosis.update(x)
    print(kurtosis.get())
```

```text
-3.0
-2.0
-1.5
-1.011599627723906
-0.9615800585356089
-0.6989395431537853
-1.4252699121794408
-1.311437071070812
-1.246289111322894
-1.082283689864171
```

```python
for i in range(2, len(X)+1):
    print(scipy.stats.kurtosis(X[:i], bias=True))
```

```text
-2.0
-1.4999999999999998
-1.0115996277239057
-0.9615800585356098
-0.6989395431537861
-1.425269912179441
-1.3114370710708125
-1.2462891113228936
-1.0822836898641714
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

- [Wikipedia article on algorithms for calculating variance](<https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Covariance>)
