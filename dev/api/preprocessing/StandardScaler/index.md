# StandardScaler

Scales the data so that it has zero mean and unit variance.

Under the hood, a running mean and a running variance are maintained. The scaling is slightly different than when scaling the data in batch because the exact means and variances are not known in advance. However, this doesn't have a detrimental impact on performance in the long run.

This transformer supports mini-batches as well as single instances. In the mini-batch case, the number of columns and the ordering of the columns are allowed to change between subsequent calls. In other words, this transformer will keep working even if you add and/or remove features every time you call `learn_many` and `transform_many`.

When `window_size` is set, the running mean and variance are replaced by rolling versions computed over the last `window_size` observations via [`utils.Rolling`](<https://riverml.xyz/dev/api/utils/Rolling>) wrapping [`stats.Mean`](<https://riverml.xyz/dev/api/stats/Mean>) and [`stats.Var`](<https://riverml.xyz/dev/api/stats/Var>). In this mode, `learn_many` is processed row by row because the mini-batch merge formula for variance does not yield a correct rolling estimate.

## Parameters

- **with\_std**

  *Default* → `True`

  Whether or not each feature should be divided by its standard deviation.
- **window\_size**

  *Type* → `int | None`

  *Default* → `None`

  Size of the rolling window used to compute the mean and variance. If `None`, the running mean and variance over the entire stream are used.

## Examples

```python
import random
from river import preprocessing

random.seed(42)
X = [{'x': random.uniform(8, 12), 'y': random.uniform(8, 12)} for _ in range(6)]
for x in X:
    print(x)
```

```text
{'x': 10.557, 'y': 8.100}
{'x': 9.100, 'y': 8.892}
{'x': 10.945, 'y': 10.706}
{'x': 11.568, 'y': 8.347}
{'x': 9.687, 'y': 8.119}
{'x': 8.874, 'y': 10.021}
```

```python
scaler = preprocessing.StandardScaler()

for x in X:
    scaler.learn_one(x)
    print(scaler.transform_one(x))
```

```text
{'x': 0.0, 'y': 0.0}
{'x': -0.999, 'y': 0.999}
{'x': 0.937, 'y': 1.350}
{'x': 1.129, 'y': -0.651}
{'x': -0.776, 'y': -0.729}
{'x': -1.274, 'y': 0.992}
```

A rolling window can be used to scale relative to the most recent observations only. The variance is the population variance (`ddof=0`), matching the running estimator used in the default mode:

```python
scaler = preprocessing.StandardScaler(window_size=3)
for x in X:
    scaler.learn_one(x)
    print(scaler.transform_one(x))
```

```text
{'x': 0.0, 'y': 0.0}
{'x': -1.0, 'y': 1.0}
{'x': 0.937, 'y': 1.351}
{'x': 0.983, 'y': -0.960}
{'x': -1.337, 'y': -0.803}
{'x': -1.036, 'y': 1.406}
```

This transformer also supports mini-batch updates. You can call `learn_many` and provide a `pandas.DataFrame`:

```python
import pandas as pd
X = pd.DataFrame.from_dict(X)

scaler = preprocessing.StandardScaler()
scaler.learn_many(X[:3])
scaler.learn_many(X[3:])
```

You can then call `transform_many` to scale a mini-batch of features:

```python
scaler.transform_many(X)
```

```text
    x         y
0  0.444600 -0.933384
1 -1.044259 -0.138809
2  0.841106  1.679208
3  1.477301 -0.685117
4 -0.444084 -0.914195
5 -1.274664  0.992296
```

A scaler can also be warm-started from previously computed statistics, e.g. to resume from a checkpoint or to seed the stream with an offline estimate:

```python
scaler = preprocessing.StandardScaler._from_state(
    counts={'x': 100},
    means={'x': 10.0},
    vars={'x': 4.0},
)
scaler.transform_one({'x': 12.0})
```

```text
{'x': 1.0}
```

## Methods

> [!NOTE]
>
> **learn\_many**
>
> Update with a mini-batch of features.
>
> Note that the update formulas for mean and variance are slightly different than in the single instance case, but they produce exactly the same result. When `window_size` is set, the rows are processed sequentially because the batched merge formula is not compatible with rolling-window eviction.
>
> **Parameters**
>
> - **X** — `IntoDataFrame`

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
> **transform\_many**
>
> Scale a mini-batch of features.
>
> Every narwhals-supported backend (pandas, polars, pyarrow, nullable/arrow-backed pandas, ...) takes the same backend-agnostic path. The compute dtype is inferred from the input schema, so a feature's float dtype is preserved (e.g. `float32` stays `float32`); integer columns are widened to the matching float and anything else falls back to `float64`.
>
> **Parameters**
>
> - **X** — `IntoDataFrameT`

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

## References

- [Welford's Method (and Friends)](<https://www.embeddedrelated.com/showarticle/785.php>)
- [Batch updates for simple statistics](<https://notmatthancock.github.io/2017/03/23/simple-batch-stat-updates.html>)
