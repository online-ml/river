# Agg

Computes a streaming aggregate.

This transformer allows to compute an aggregate statistic, very much like the groupby method from `pandas`, but on a streaming dataset. This makes use of the streaming statistics from the `stats` module.

When `learn_one` is called, the running statistic `how` of group `by` is updated with the value of `on`. Meanwhile, the output of `transform_one` is a single-element dictionary, where the key is the name of the aggregate and the value is the current value of the statistic for the relevant group. The key is automatically inferred from the parameters.

Note that you can use a [`compose.TransformerUnion`](<https://riverml.xyz/dev/api/compose/TransformerUnion>) to extract many aggregate statistics in a concise manner.

## Parameters

- **on**

  *Type* → `str`

  The feature on which to compute the aggregate statistic.
- **by**

  *Type* → `str | list[str] | None`

  The feature by which to group the data. All the data is included in the aggregate if this is `None`.
- **how**

  *Type* → `stats.base.Univariate[typing.Any] | utils.Rolling | utils.TimeRolling`

  The statistic to compute.

## Attributes

- **state**

  Return the current values for each group as a series.

## Examples

Consider the following dataset:

```python
X = [
    {'country': 'France', 'place': 'Taco Bell', 'revenue': 42},
    {'country': 'Sweden', 'place': 'Burger King', 'revenue': 16},
    {'country': 'France', 'place': 'Burger King', 'revenue': 24},
    {'country': 'Sweden', 'place': 'Taco Bell', 'revenue': 58},
    {'country': 'Sweden', 'place': 'Burger King', 'revenue': 20},
    {'country': 'France', 'place': 'Taco Bell', 'revenue': 50},
    {'country': 'France', 'place': 'Burger King', 'revenue': 10},
    {'country': 'Sweden', 'place': 'Taco Bell', 'revenue': 80}
]
```

As an example, we can calculate the average (how) revenue (on) for each place (by):

```python
from river import feature_extraction as fx
from river import stats

agg = fx.Agg(
    on='revenue',
    by='place',
    how=stats.Mean()
)

for x in X:
    agg.learn_one(x)
    print(agg.transform_one(x))
```

```text
{'revenue_mean_by_place': 42.0}
{'revenue_mean_by_place': 16.0}
{'revenue_mean_by_place': 20.0}
{'revenue_mean_by_place': 50.0}
{'revenue_mean_by_place': 20.0}
{'revenue_mean_by_place': 50.0}
{'revenue_mean_by_place': 17.5}
{'revenue_mean_by_place': 57.5}
```

You can compute an aggregate over multiple keys by passing a tuple to the `by` argument. For instance, we can compute the maximum (how) revenue (on) per place as well as per day (by):

```python
agg = fx.Agg(
    on='revenue',
    by=['place', 'country'],
    how=stats.Max()
)

for x in X:
    agg.learn_one(x)
    print(agg.transform_one(x))
```

```text
{'revenue_max_by_place_and_country': 42}
{'revenue_max_by_place_and_country': 16}
{'revenue_max_by_place_and_country': 24}
{'revenue_max_by_place_and_country': 58}
{'revenue_max_by_place_and_country': 20}
{'revenue_max_by_place_and_country': 50}
{'revenue_max_by_place_and_country': 24}
{'revenue_max_by_place_and_country': 80}
```

You can use a [`compose.TransformerUnion`](<https://riverml.xyz/dev/api/compose/TransformerUnion>) in order to calculate multiple aggregates in one go. The latter can be constructed by using the `+` operator:

```python
agg = (
    fx.Agg(on='revenue', by='place', how=stats.Mean()) +
    fx.Agg(on='revenue', by=['place', 'country'], how=stats.Max())
)

import pprint
for x in X:
    agg.learn_one(x)
    pprint.pprint(agg.transform_one(x))
```

```text
{'revenue_max_by_place_and_country': 42, 'revenue_mean_by_place': 42.0}
{'revenue_max_by_place_and_country': 16, 'revenue_mean_by_place': 16.0}
{'revenue_max_by_place_and_country': 24, 'revenue_mean_by_place': 20.0}
{'revenue_max_by_place_and_country': 58, 'revenue_mean_by_place': 50.0}
{'revenue_max_by_place_and_country': 20, 'revenue_mean_by_place': 20.0}
{'revenue_max_by_place_and_country': 50, 'revenue_mean_by_place': 50.0}
{'revenue_max_by_place_and_country': 24, 'revenue_mean_by_place': 17.5}
{'revenue_max_by_place_and_country': 80, 'revenue_mean_by_place': 57.5}
```

The `state` property returns a `pandas.Series`, which can be useful for visualizing the current state.

```python
agg[0].state
```

```text
Taco Bell      57.5
Burger King    17.5
Name: revenue_mean_by_place, dtype: float64
```

```python
agg[1].state
```

```text
place        country
Taco Bell    France     50
Burger King  Sweden     20
             France     24
Taco Bell    Sweden     80
Name: revenue_max_by_place_and_country, dtype: int64
```

This transformer can also be used in conjunction with [`utils.TimeRolling`](<https://riverml.xyz/dev/api/utils/TimeRolling>). The latter requires a `t` argument, which is a timestamp that indicates when the current row was observed. For instance, we can calculate the average (how) revenue (on) for each place (by) over the last 7 days (t):

```python
import datetime as dt
import random
import string
from river import utils

agg = fx.Agg(
    on="value",
    by="group",
    how=utils.TimeRolling(stats.Mean, dt.timedelta(days=7))
)

for day in range(366):
    g = random.choice(string.ascii_lowercase)
    x = {
        "group": g,
        "value": string.ascii_lowercase.index(g) + random.random(),
    }
    t = dt.datetime(2023, 1, 1) + dt.timedelta(days=day)
    agg.learn_one(x, t=t)

len(agg.state)
```

```text
26
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
> - **t** — defaults to `None`

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

- [Streaming groupbys in pandas for big datasets](<https://maxhalford.github.io/blog/pandas-streaming-groupby/>)
