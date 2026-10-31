# MaxAbsScaler

Scales the data to a \[-1, 1\] range based on absolute maximum.

Under the hood a running absolute max is maintained. This scaler is meant for data that is already centered at zero or sparse data. It does not shift/center the data, and thus does not destroy any sparsity.

When `window_size` is set, the scaler tracks the absolute max over the last `window_size` observations via [`stats.RollingAbsMax`](<https://riverml.xyz/dev/api/stats/RollingAbsMax>) instead of the entire stream.

## Parameters

- **window\_size**

  *Type* → `int | None`

  *Default* → `None`

  Size of the rolling window used to compute the absolute max. If `None`, the running absolute max over the entire stream is used.

## Attributes

- **abs\_max** (`dict`)

  Mapping between features and instances of `stats.AbsMax` (or `stats.RollingAbsMax` when `window_size` is set).

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
scaler = preprocessing.MaxAbsScaler()

for x in X:
    scaler.learn_one(x)
    print(scaler.transform_one(x))
```

```text
{'x': 1.0}
{'x': 0.767216}
{'x': 0.861940}
{'x': 0.842308}
{'x': 1.0}
```

A scaler can also be warm-started from a previously computed absolute max:

```python
scaler = preprocessing.MaxAbsScaler._from_state(abs_max={'x': 12.0})
scaler.transform_one({'x': 6.0})
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
