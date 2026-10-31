# MinMaxScaler

Scales the data to a fixed range from 0 to 1.

Under the hood a running min and a running peak to peak (max - min) are maintained. When `window_size` is set, the scaler tracks the min and max over the last `window_size` observations via [`stats.RollingMin`](<https://riverml.xyz/dev/api/stats/RollingMin>) and [`stats.RollingMax`](<https://riverml.xyz/dev/api/stats/RollingMax>) instead.

## Parameters

- **window\_size**

  *Type* → `int | None`

  *Default* → `None`

  Size of the rolling window used to compute the min and max. If `None`, the running min and max over the entire stream are used.

## Attributes

- **min** (`dict`)

  Mapping between features and instances of `stats.Min` (or `stats.RollingMin` when `window_size` is set).
- **max** (`dict`)

  Mapping between features and instances of `stats.Max` (or `stats.RollingMax` when `window_size` is set).

## Examples

```python
import random
from river import preprocessing

random.seed(42)
X = [{'x': random.uniform(8, 12)} for _ in range(5)]
for x in X:
    print(x)
```

```text
{'x': 10.557707}
{'x': 8.100043}
{'x': 9.100117}
{'x': 8.892842}
{'x': 10.945884}
```

```python
scaler = preprocessing.MinMaxScaler()

for x in X:
    scaler.learn_one(x)
    print(scaler.transform_one(x))
```

```text
{'x': 0.0}
{'x': 0.0}
{'x': 0.406920}
{'x': 0.322582}
{'x': 1.0}
```

A rolling window can be used to scale relative to the most recent observations only:

```python
scaler = preprocessing.MinMaxScaler(window_size=3)
for x in X:
    scaler.learn_one(x)
    print(scaler.transform_one(x))
```

```text
{'x': 0.0}
{'x': 0.0}
{'x': 0.406920}
{'x': 0.792741}
{'x': 1.0}
```

A scaler can also be warm-started from previously computed statistics, e.g. to resume from a checkpoint or to seed the stream with an offline estimate:

```python
scaler = preprocessing.MinMaxScaler._from_state(min={'x': 8.0}, max={'x': 12.0})
scaler.transform_one({'x': 10.0})
```

```text
{'x': 0.5}
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
