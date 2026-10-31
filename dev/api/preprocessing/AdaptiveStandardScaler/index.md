# AdaptiveStandardScaler

Scales data using exponentially weighted moving average and variance.

Under the hood, a exponentially weighted running mean and variance are maintained for each feature. This can potentially provide better results for drifting data in comparison to [`preprocessing.StandardScaler`](<https://riverml.xyz/dev/api/preprocessing/StandardScaler>). Indeed, the latter computes a global mean and variance for each feature, whereas this scaler weights data in proportion to their recency.

## Parameters

- **fading\_factor**

  *Default* → `0.3`

  This parameter is passed to `stats.EWVar`. It is expected to be in \[0, 1\]. More weight is assigned to recent samples the closer `fading_factor` is to 1.

## Examples

Consider the following series which contains a positive trend.

```python
import random

random.seed(42)
X = [
    {'x': random.uniform(4 + i, 6 + i)}
    for i in range(8)
]
for x in X:
    print(x)
```

```text
{'x': 5.278}
{'x': 5.050}
{'x': 6.550}
{'x': 7.446}
{'x': 9.472}
{'x': 10.353}
{'x': 11.784}
{'x': 11.173}
```

This scaler works well with this kind of data because it uses statistics that assign higher weight to more recent data.

```python
from river import preprocessing

scaler = preprocessing.AdaptiveStandardScaler(fading_factor=.6)

for x in X:
    scaler.learn_one(x)
    print(scaler.transform_one(x))
```

```text
{'x': 0.0}
{'x': -0.816}
{'x': 0.812}
{'x': 0.695}
{'x': 0.754}
{'x': 0.598}
{'x': 0.651}
{'x': 0.124}
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
