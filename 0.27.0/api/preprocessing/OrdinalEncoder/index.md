# OrdinalEncoder

Ordinal encoder.

This transformer maps each feature to integers. It can useful when a feature has string values (i.e. categorical variables).

## Parameters

- **categories**

  *Type* → `dict[base.typing.FeatureName, dict[typing.Hashable, int]] | None`

  *Default* → `None`

  Categories (unique values) per feature: `None` : Determine categories automatically from the training data. dict of dicts : Expected categories for each feature. The outer dict maps each feature to its inner dict. The inner dict maps each category to its code. The used categories can be found in the `values` attribute.
- **unknown\_value**

  *Type* → `int | None`

  *Default* → `0`

  The value to use for unknown categories seen during `transform_one`. Unknown categories will be mapped to an integer once they are seen during `learn_one`. This value can be set to `None` in order to categories to `None` if they've never been seen before.
- **none\_value**

  *Type* → `int`

  *Default* → `-1`

  The value to encode `None` with.

## Attributes

- **values**

  A dict of dicts. The outer dict maps each feature to its inner dict. The inner dict maps each category to its code.

## Examples

```python
from river import preprocessing

X = [
    {"country": "France", "place": "Taco Bell"},
    {"country": None, "place": None},
    {"country": "Sweden", "place": "Burger King"},
    {"country": "France", "place": "Burger King"},
    {"country": "Russia", "place": "Starbucks"},
    {"country": "Russia", "place": "Starbucks"},
    {"country": "Sweden", "place": "Taco Bell"},
    {"country": None, "place": None},
]

encoder = preprocessing.OrdinalEncoder()
for x in X:
    print(encoder.transform_one(x))
    encoder.learn_one(x)
```

```text
{'country': 0, 'place': 0}
{'country': -1, 'place': -1}
{'country': 0, 'place': 0}
{'country': 1, 'place': 2}
{'country': 0, 'place': 0}
{'country': 3, 'place': 3}
{'country': 2, 'place': 1}
{'country': -1, 'place': -1}
```

Like in `scikit-learn`, you can also specify the expected categories manually. This is handy when you want to constrain category encoding space to e.g. top 20% most popular category values you've picked in advance.

```python
categories = {'country': {'France': 1},
              'place': {'Burger King': 2, 'Starbucks': 3}}
encoder = preprocessing.OrdinalEncoder(categories=categories)
for x in X:
    print(encoder.transform_one(x))
    encoder.learn_one(x)
```

```text
{'country': 1, 'place': 0}
{'country': -1, 'place': -1}
{'country': 0, 'place': 2}
{'country': 1, 'place': 2}
{'country': 0, 'place': 3}
{'country': 0, 'place': 3}
{'country': 0, 'place': 0}
{'country': -1, 'place': -1}
```

```python
import pandas as pd
xb1 = pd.DataFrame(X[0:4], index=[0, 1, 2, 3])
xb2 = pd.DataFrame(X[4:8], index=[4, 5, 6, 7])

encoder = preprocessing.OrdinalEncoder()
encoder.transform_many(xb1)
```

```text
   country  place
0        0      0
1       -1     -1
2        0      0
3        0      0
```

```python
encoder.learn_many(xb1)
encoder.transform_many(xb2)
```

```text
   country  place
4        0      0
5        0      0
6        2      1
7       -1     -1
```

## Methods

> [!NOTE]
>
> **learn\_many**
>
> Update with a mini-batch of features.
>
> A lot of transformers don't actually have to do anything during the `learn_many` step because they are stateless. For this reason the default behavior of this function is to do nothing. Transformers that however do something during the `learn_many` can override this method.
>
> **Parameters**
>
> - **X** — `IntoDataFrame`
> - **y** — `IntoSeries | None` — defaults to `None`

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
> Encode a mini-batch of features into integer codes.
>
> Missing cells (`None`/`NaN`/`pd.NA`) are mapped to `none_value` consistently across backends; values absent from the learned categories map to `unknown_value`.
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
