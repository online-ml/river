# TargetAgg

Computes a streaming aggregate of the target values.

This transformer is identical to [`feature_extraction.Agg`](<https://riverml.xyz/dev/api/feature-extraction/Agg>), the only difference is that it operates on the target rather than on a feature. At each step, the running statistic `how` of target values in group `by` is updated with the target. It is therefore a supervised transformer.

## Parameters

- **by**

  *Type* → `str | list[str] | None`

  The feature by which to group the target values. All the data is included in the aggregate if this is `None`.
- **how**

  *Type* → `stats.base.Univariate[typing.Any] | utils.Rolling | utils.TimeRolling`

  The statistic to compute.
- **target\_name**

  *Default* → `y`

  The target name which is used in the result.

## Attributes

- **state**

  Return the current values for each group as a series.
- **target\_name**

## Examples

Consider the following dataset, where the second value of each value is the target:

```python
dataset = [
    ({'country': 'France', 'place': 'Taco Bell'}, 42),
    ({'country': 'Sweden', 'place': 'Burger King'}, 16),
    ({'country': 'France', 'place': 'Burger King'}, 24),
    ({'country': 'Sweden', 'place': 'Taco Bell'}, 58),
    ({'country': 'Sweden', 'place': 'Burger King'}, 20),
    ({'country': 'France', 'place': 'Taco Bell'}, 50),
    ({'country': 'France', 'place': 'Burger King'}, 10),
    ({'country': 'Sweden', 'place': 'Taco Bell'}, 80)
]
```

As an example, let's perform a target encoding of the `place` feature. Instead of simply updating a running average, we use a [`stats.BayesianMean`](<https://riverml.xyz/dev/api/stats/BayesianMean>) which allows us to incorporate some prior knowledge. This makes subsequent models less prone to overfitting. Indeed, it dampens the fact that too few samples might have been seen within a group.

```python
from river import feature_extraction
from river import stats

agg = feature_extraction.TargetAgg(
    by='place',
    how=stats.BayesianMean(
        prior=3,
        prior_weight=1
    )
)

for x, y in dataset:
    print(agg.transform_one(x))
    agg.learn_one(x, y)
```

```text
{'y_bayes_mean_by_place': 3.0}
{'y_bayes_mean_by_place': 3.0}
{'y_bayes_mean_by_place': 9.5}
{'y_bayes_mean_by_place': 22.5}
{'y_bayes_mean_by_place': 14.333}
{'y_bayes_mean_by_place': 34.333}
{'y_bayes_mean_by_place': 15.75}
{'y_bayes_mean_by_place': 38.25}
```

Just like with [`feature_extraction.Agg`](<https://riverml.xyz/dev/api/feature-extraction/Agg>), we can specify multiple features on which to group the data:

```python
agg = feature_extraction.TargetAgg(
    by=['place', 'country'],
    how=stats.BayesianMean(
        prior=3,
        prior_weight=1
    )
)

for x, y in dataset:
    print(agg.transform_one(x))
    agg.learn_one(x, y)
```

```text
{'y_bayes_mean_by_place_and_country': 3.0}
{'y_bayes_mean_by_place_and_country': 3.0}
{'y_bayes_mean_by_place_and_country': 3.0}
{'y_bayes_mean_by_place_and_country': 3.0}
{'y_bayes_mean_by_place_and_country': 9.5}
{'y_bayes_mean_by_place_and_country': 22.5}
{'y_bayes_mean_by_place_and_country': 13.5}
{'y_bayes_mean_by_place_and_country': 30.5}
```

```python
agg.state
```

```text
place        country
Taco Bell    France     31.666667
Burger King  Sweden     13.000000
             France     12.333333
Taco Bell    Sweden     47.000000
Name: y_bayes_mean_by_place_and_country, dtype: float64
```

This transformer can also be used in conjunction with [`utils.TimeRolling`](<https://riverml.xyz/dev/api/utils/TimeRolling>). The latter requires a `t` argument, which is a timestamp that indicates when the current row was observed. For instance, we can calculate the average (how) revenue (on) for each place (by) over the last 7 days (t):

```python
import datetime as dt
import random
import string
from river import utils

agg = feature_extraction.TargetAgg(
    by="group",
    how=utils.TimeRolling(stats.Mean, dt.timedelta(days=7))
)

for day in range(366):
    g = random.choice(string.ascii_lowercase)
    x = {"group": g}
    y = string.ascii_lowercase.index(g) + random.random()
    t = dt.datetime(2023, 1, 1) + dt.timedelta(days=day)
    agg.learn_one(x, y, t=t)
```

## Methods

> [!NOTE]
>
> **learn\_one**
>
> Update with a set of features `x` and a target `y`.
>
> **Parameters**
>
> - **x** — `dict[base.typing.FeatureName, Any]`
> - **y** — `base.typing.Target`
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

- 1. [Streaming groupbys in pandas for big datasets](<https://maxhalford.github.io/blog/streaming-groupbys-in-pandas-for-big-datasets/>)
