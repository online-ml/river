# StatImputer

Replaces missing values with a statistic.

This transformer allows you to replace missing values with the value of a running statistic. During a call to `learn_one`, for each feature, a statistic is updated whenever a numeric feature is observed. When `transform_one` is called, each feature with a `None` value is replaced with the current value of the corresponding statistic.

## Examples

```python
from river import preprocessing
from river import stats
```

For numeric data, we can use a [`stats.Mean`](<https://riverml.xyz/dev/api/stats/Mean>)()\` to replace missing values by the running average of the previously seen values:

```python
X = [
    {'temperature': 1},
    {'temperature': 8},
    {'temperature': 3},
    {'temperature': None},
    {'temperature': 4}
]

imp = preprocessing.StatImputer(('temperature', stats.Mean()))

for x in X:
    imp.learn_one(x)
    print(imp.transform_one(x))
```

```text
{'temperature': 1}
{'temperature': 8}
{'temperature': 3}
{'temperature': 4.0}
{'temperature': 4}
```

For discrete/categorical data, a common practice is to [`stats.Mode`](<https://riverml.xyz/dev/api/stats/Mode>) to replace missing values by the most commonly seen value:

```python
X = [
    {'weather': 'sunny'},
    {'weather': 'rainy'},
    {'weather': 'sunny'},
    {'weather': None},
    {'weather': 'rainy'},
    {'weather': 'rainy'},
    {'weather': None}
]

imp = preprocessing.StatImputer(('weather', stats.Mode()))

for x in X:
    imp.learn_one(x)
    print(imp.transform_one(x))
```

```text
{'weather': 'sunny'}
{'weather': 'rainy'}
{'weather': 'sunny'}
{'weather': 'sunny'}
{'weather': 'rainy'}
{'weather': 'rainy'}
{'weather': 'rainy'}
```

You can also choose to replace missing values with a constant value, as so:

```python
imp = preprocessing.StatImputer(('weather', 'missing'))

for x in X:
    imp.learn_one(x)
    print(imp.transform_one(x))
```

```text
{'weather': 'sunny'}
{'weather': 'rainy'}
{'weather': 'sunny'}
{'weather': 'missing'}
{'weather': 'rainy'}
{'weather': 'rainy'}
{'weather': 'missing'}
```

Multiple imputers can be defined by providing a tuple for each feature which you want to impute:

```python
X = [
    {'weather': 'sunny', 'temperature': 8},
    {'weather': 'rainy', 'temperature': 3},
    {'weather': 'sunny', 'temperature': None},
    {'weather': None, 'temperature': 4},
    {'weather': 'snowy', 'temperature': -4},
    {'weather': 'snowy', 'temperature': -3},
    {'weather': 'snowy', 'temperature': -3},
    {'weather': None, 'temperature': None}
]

imp = preprocessing.StatImputer(
    ('temperature', stats.Mean()),
    ('weather', stats.Mode())
)

for x in X:
    imp.learn_one(x)
    print(imp.transform_one(x))
```

```text
{'weather': 'sunny', 'temperature': 8}
{'weather': 'rainy', 'temperature': 3}
{'weather': 'sunny', 'temperature': 5.5}
{'weather': 'sunny', 'temperature': 4}
{'weather': 'snowy', 'temperature': -4}
{'weather': 'snowy', 'temperature': -3}
{'weather': 'snowy', 'temperature': -3}
{'weather': 'snowy', 'temperature': 0.8333}
```

A sophisticated way to go about imputation is condition the statistics on a given feature. For instance, we might want to replace a missing temperature with the average temperature of a particular weather condition. As an example, consider the following dataset where the temperature is missing, but not the weather condition:

```python
X = [
    {'weather': 'sunny', 'temperature': 8},
    {'weather': 'rainy', 'temperature': 3},
    {'weather': 'sunny', 'temperature': None},
    {'weather': 'rainy', 'temperature': 4},
    {'weather': 'sunny', 'temperature': 10},
    {'weather': 'sunny', 'temperature': None},
    {'weather': 'sunny', 'temperature': 12},
    {'weather': 'rainy', 'temperature': None}
]
```

Each missing temperature can be replaced with the average temperature of the corresponding weather condition as so:

```python
from river import compose

imp = compose.Grouper(
    preprocessing.StatImputer(('temperature', stats.Mean())),
    by='weather'
)

for x in X:
    imp.learn_one(x)
    print(imp.transform_one(x))
```

```text
{'weather': 'sunny', 'temperature': 8}
{'weather': 'rainy', 'temperature': 3}
{'weather': 'sunny', 'temperature': 8.0}
{'weather': 'rainy', 'temperature': 4}
{'weather': 'sunny', 'temperature': 10}
{'weather': 'sunny', 'temperature': 9.0}
{'weather': 'sunny', 'temperature': 12}
{'weather': 'rainy', 'temperature': 3.5}
```

Note that you can also create a `Grouper` with the `*` operator:

```python
imp = preprocessing.StatImputer(('temperature', stats.Mean())) * 'weather'
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
