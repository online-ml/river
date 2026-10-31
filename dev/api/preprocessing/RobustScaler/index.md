# RobustScaler

Scale features using statistics that are robust to outliers.

This Scaler removes the median and scales the data according to the interquantile range.

## Parameters

- **with\_centering**

  *Default* → `True`

  Whether to centre the data before scaling.
- **with\_scaling**

  *Default* → `True`

  Whether to scale data to IQR.
- **q\_inf**

  *Default* → `0.25`

  Desired inferior quantile, must be between 0 and 1.
- **q\_sup**

  *Default* → `0.75`

  Desired superior quantile, must be between 0 and 1.

## Attributes

- **median** (`dict`)

  Mapping between features and instances of `stats.Quantile(0.5)`.
- **iqr** (`dict`)

  Mapping between features and instances of `stats.IQR`.

## Examples

```python
from pprint import pprint
import random
from river import preprocessing

random.seed(42)
X = [{'x': random.uniform(8, 12)} for _ in range(5)]
pprint(X)
```

```text
[{'x': 10.557707},
    {'x': 8.100043},
    {'x': 9.100117},
    {'x': 8.892842},
    {'x': 10.945884}]
```

```python
scaler = preprocessing.RobustScaler()

for x in X:
    scaler.learn_one(x)
    print(scaler.transform_one(x))
```

```text
    {'x': 0.0}
    {'x': -1.0}
    {'x': 0.0}
    {'x': -0.12449923287875722}
    {'x': 1.1086595155704708}
```

## Methods

> [!NOTE]
>
> **learn\_one**
>
> Update with a set of features `x`.
>
> A lot of transformers don't actually have to do anything during the `learn_one` step because they are stateless. For this reason the default behavior of this function is to do nothing. Transformers that however do something during the `learn_one` can override this method.
>
> **Parameters**
>
> - **x** — `dict[base.typing.FeatureName, Any]`

> [!NOTE]
>
> **transform\_one**
>
> Transform a set of features `x`.
>
> **Parameters**
>
> - **x** — `dict[base.typing.FeatureName, Any]`
>
> **Returns**
>
> `dict&#91;base.typing.FeatureName, Any&#93;`: The transformed values.
