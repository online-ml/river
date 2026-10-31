# Shift

Shifts a data stream by returning past values.

This can be used to compute statistics over past data. For instance, if you're computing daily averages, then shifting by 7 will be equivalent to computing averages from a week ago.

Shifting values is useful when you're calculating an average over a target value. Indeed, in this case it's important to shift the values in order not to introduce leakage. The recommended way to do this is to [`feature_extraction.TargetAgg`](<https://riverml.xyz/dev/api/feature-extraction/TargetAgg>), which already takes care of shifting the target values once.

## Parameters

- **amount**

  *Type* → `int`

  *Default* → `1`

  Shift amount. The `get` method will return the `t - amount` value, where `t` is the current moment.
- **fill\_value**

  *Type* → `float | None`

  *Default* → `None`

  This value will be returned by the `get` method if not enough values have been observed.

## Attributes

- **name**

## Examples

It is rare to have to use `Shift` by itself. A more common usage is to compose it with other statistics. This can be done via the `|` operator.

```python
from river import stats

stat = stats.Shift(1) | stats.Mean()

for i in range(5):
    stat.update(i)
    print(stat.get())
```

```text
0.0
0.0
0.5
1.0
1.5
```

A common usecase for using `Shift` is when computing statistics on shifted data. For instance, say you have a dataset which records the amount of sales for a set of shops. You might then have a `shop` field and a `sales` field. Let's say you want to look at the average amount of sales per shop. You can do this by using a [`feature_extraction.Agg`](<https://riverml.xyz/dev/api/feature-extraction/Agg>). When you call `transform_one`, you're expecting it to return the average amount of sales, *without* including today's sales. You can do this by prepending an instance of [`stats.Mean`](<https://riverml.xyz/dev/api/stats/Mean>) with an instance of [`stats.Shift`](<https://riverml.xyz/dev/api/stats/Shift>).

```python
from river import feature_extraction

agg = feature_extraction.Agg(
    on='sales',
    how=stats.Shift(1) | stats.Mean(),
    by='shop'
)
```

Let's define a little example dataset.

```python
X = iter([
    {'shop': 'Ikea', 'sales': 10},
    {'shop': 'Ikea', 'sales': 15},
    {'shop': 'Ikea', 'sales': 20}
])
```

Now let's call the `learn_one` method to update our feature extractor.

```python
x = next(X)
agg.learn_one(x)
```

At this point, the average defaults to the initial value of [`stats.Mean`](<https://riverml.xyz/dev/api/stats/Mean>), which is 0.

```python
agg.transform_one(x)
```

```text
{'sales_mean_of_shift_1_by_shop': 0.0}
```

We can now update our feature extractor with the next data point and check the output.

```python
agg.learn_one(next(X))
agg.transform_one(x)
```

```text
{'sales_mean_of_shift_1_by_shop': 10.0}
```

```python
agg.learn_one(next(X))
agg.transform_one(x)
```

```text
{'sales_mean_of_shift_1_by_shop': 12.5}
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
