# Skew

Running skew using Welford's algorithm.

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
import numpy as np

np.random.seed(42)
X = np.random.normal(loc=0, scale=1, size=10)

skew = stats.Skew(bias=False)
for x in X:
    skew.update(x)
    print(skew.get())
```

```text
0.0
0.0
-1.4802398132849872
0.5127437186677888
0.7803466510704751
1.056115628922055
0.5057840774320389
0.3478402420400934
0.4536710660918703
0.41230701974932277
```

```python
skew = stats.Skew(bias=True)
for x in X:
    skew.update(x)
    print(skew.get())
```

```text
0.0
0.0
-0.6043053732501439
0.2960327239981376
0.5234724473423674
0.7712778043924866
0.39022088752624845
0.278892645224261
0.3742595351386406
0.34768780738236965
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
