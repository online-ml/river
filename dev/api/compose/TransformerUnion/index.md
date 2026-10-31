# TransformerUnion

Packs multiple transformers into a single one.

Pipelines allow you to apply steps sequentially. Therefore, the output of a step becomes the input of the next one. In many cases, you may want to pass the output of a step to multiple steps. This simple transformer allows you to do so. In other words, it enables you to apply particular steps to different parts of an input. A typical example is when you want to scale numeric features and one-hot encode categorical features.

This transformer is essentially a list of transformers. Whenever it is updated, it loops through each transformer and updates them. Meanwhile, calling `transform_one` collects the output of each transformer and merges them into a single dictionary.

## Examples

Take the following dataset:

```python
X = [
    {'place': 'Taco Bell', 'revenue': 42},
    {'place': 'Burger King', 'revenue': 16},
    {'place': 'Burger King', 'revenue': 24},
    {'place': 'Taco Bell', 'revenue': 58},
    {'place': 'Burger King', 'revenue': 20},
    {'place': 'Taco Bell', 'revenue': 50}
]
```

As an example, let's assume we want to compute two aggregates of a dataset. We therefore define two [`feature_extraction.Agg`](<https://riverml.xyz/dev/api/feature-extraction/Agg>)s and initialize a `TransformerUnion` with them:

```python
import pandas as pd
from river import compose
from river import feature_extraction
from river import stats

mean = feature_extraction.Agg(
    on='revenue', by='place',
    how=stats.Mean()
)
count = feature_extraction.Agg(
    on='revenue', by='place',
    how=stats.Count()
)
agg = compose.TransformerUnion(mean, count)
```

We can now update each transformer and obtain their output with a single function call:

```python
from pprint import pprint
for x in X:
    agg.learn_one(x)
    pprint(agg.transform_one(x))
```

```text
{'revenue_count_by_place': 1, 'revenue_mean_by_place': 42.0}
{'revenue_count_by_place': 1, 'revenue_mean_by_place': 16.0}
{'revenue_count_by_place': 2, 'revenue_mean_by_place': 20.0}
{'revenue_count_by_place': 2, 'revenue_mean_by_place': 50.0}
{'revenue_count_by_place': 3, 'revenue_mean_by_place': 20.0}
{'revenue_count_by_place': 3, 'revenue_mean_by_place': 50.0}
```

Note that you can use the `+` operator as a shorthand notation:

agg = mean + count

This allows you to build complex pipelines in a very terse manner. For instance, we can create a pipeline that scales each feature and fits a logistic regression as so:

```python
from river import linear_model as lm
from river import preprocessing as pp

model = (
    (mean + count) |
    pp.StandardScaler() |
    lm.LogisticRegression()
)
```

Whice is equivalent to the following code:

```python
model = compose.Pipeline(
    compose.TransformerUnion(mean, count),
    pp.StandardScaler(),
    lm.LogisticRegression()
)
```

Note that you access any part of a `TransformerUnion` by name:

```python
model['TransformerUnion']['Agg']
```

```text
Agg (
    on="revenue"
    by=['place']
    how=Mean ()
)
```

```python
model['TransformerUnion']['Agg1']
```

```text
Agg (
    on="revenue"
    by=['place']
    how=Count ()
)
```

You can also manually provide a name for each step:

```python
agg = compose.TransformerUnion(
    ('Mean revenue by place', mean),
    ('# by place', count)
)
```

Mini-batch example:

```python
X = pd.DataFrame([
    {"place": 2, "revenue": 42},
    {"place": 3, "revenue": 16},
    {"place": 3, "revenue": 24},
    {"place": 2, "revenue": 58},
    {"place": 3, "revenue": 20},
    {"place": 2, "revenue": 50},
])
```

Since we need a transformer with mini-batch support to demonstrate, we shall use a `StandardScaler`.

```python
from river import compose
from river import preprocessing

agg = (
    compose.Select("place") +
    (compose.Select("revenue") | preprocessing.StandardScaler())
)

agg.learn_many(X)
agg.transform_many(X)
```

```text
   place   revenue
0      2  0.441250
1      3 -1.197680
2      3 -0.693394
3      2  1.449823
4      3 -0.945537
5      2  0.945537
```

## Methods

> [!NOTE]
>
> **learn\_many**
>
> Update each transformer.
>
> **Parameters**
>
> - **X** — `IntoDataFrame`
> - **y** — `IntoSeries | None` — defaults to `None`

> [!NOTE]
>
> **learn\_one**
>
> Update each transformer.
>
> **Parameters**
>
> - **x** — `dict[base.typing.FeatureName, Any]`
> - **y** — defaults to `None`

> [!NOTE]
>
> **transform\_many**
>
> Passes the data through each transformer and packs the results together.
>
> The per-transformer outputs are concatenated column-wise via narwhals, so the result is rebuilt in the caller's own backend (pandas, polars, pyarrow, ...). All transformers see the same rows in the same order, so a positional horizontal concat matches the historical index-aligned `pandas.concat(axis=1)`.
>
> **Parameters**
>
> - **X** — `pd.DataFrame`

> [!NOTE]
>
> **transform\_one**
>
> Passes the data through each transformer and packs the results together.
>
> **Parameters**
>
> - **x** — `dict[base.typing.FeatureName, Any]`
