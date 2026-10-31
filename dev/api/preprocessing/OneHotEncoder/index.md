# OneHotEncoder

One-hot encoding.

This transformer will encode every feature it is provided with. If a list or set is provided, this transformer will encode every entry in the list/set. You can apply it to a subset of features by composing it with [`compose.Select`](<https://riverml.xyz/dev/api/compose/Select>) or [`compose.SelectType`](<https://riverml.xyz/dev/api/compose/SelectType>).

## Parameters

- **categories**

  *Type* → `dict[base.typing.FeatureName, set[typing.Hashable]] | None`

  *Default* → `None`

  Categories (unique values) per feature: `None` : Determine categories automatically from the training data. dict of sets : Expected categories for each feature. The outer dict maps each feature to the set of category values to encode. The used categories can be found in the `values` attribute.
- **drop\_zeros**

  *Type* → `bool`

  *Default* → `False`

  Whether or not 0s should be made explicit or not.
- **drop\_first**

  *Type* → `bool`

  *Default* → `False`

  Whether to get `k - 1` dummies out of `k` categorical levels by removing the first key. This is useful in some statistical models where perfectly collinear features cause problems.

## Attributes

- **values**

  A dict of sets. Maps each feature to the set of category values seen during training (or the categories provided explicitly).

## Examples

Let us first create an example dataset.

```python
from pprint import pprint
import random
import string

random.seed(42)
alphabet = list(string.ascii_lowercase)
X = [
    {
        'c1': random.choice(alphabet),
        'c2': random.choice(alphabet),
    }
    for _ in range(4)
]
pprint(X)
```

```text
[{'c1': 'u', 'c2': 'd'},
 {'c1': 'a', 'c2': 'x'},
 {'c1': 'i', 'c2': 'h'},
 {'c1': 'h', 'c2': 'e'}]
```

We can now apply one-hot encoding. All the provided are one-hot encoded, there is therefore no need to specify which features to encode.

```python
from river import preprocessing

oh = preprocessing.OneHotEncoder()
for x in X[:2]:
    oh.learn_one(x)
    pprint(oh.transform_one(x))
```

```text
{'c1_u': 1, 'c2_d': 1}
{'c1_a': 1, 'c1_u': 0, 'c2_d': 0, 'c2_x': 1}
```

The `drop_zeros` parameter can be set to `True` if you don't want the past features to be included in the output. Otherwise, all the past features will be included in the output.

```python
oh = preprocessing.OneHotEncoder(drop_zeros=True)
for x in X:
    oh.learn_one(x)
    pprint(oh.transform_one(x))
```

```text
{'c1_u': 1, 'c2_d': 1}
{'c1_a': 1, 'c2_x': 1}
{'c1_i': 1, 'c2_h': 1}
{'c1_h': 1, 'c2_e': 1}
```

You can encode only `k - 1` features out of `k` by setting `drop_first` to `True`.

```python
oh = preprocessing.OneHotEncoder(drop_first=True, drop_zeros=True)
for x in X:
    oh.learn_one(x)
    pprint(oh.transform_one(x))
```

```text
{'c2_d': 1}
{'c2_x': 1}
{'c1_i': 1, 'c2_h': 1}
{'c1_h': 1, 'c2_e': 1}
```

Like in `scikit-learn`, you can also specify the expected categories manually. This is handy when you want to constrain category encoding space to e.g. top 20% most popular category values you've picked in advance.

```python
categories = {'c1': {'a', 'h'}, 'c2': {'x', 'e'}}
oh = preprocessing.OneHotEncoder(categories=categories)
for x in X:
    oh.learn_one(x)
    pprint(oh.transform_one(x))
```

```text
{'c1_a': 0, 'c1_h': 0, 'c2_e': 0, 'c2_x': 0}
{'c1_a': 1, 'c1_h': 0, 'c2_e': 0, 'c2_x': 1}
{'c1_a': 0, 'c1_h': 0, 'c2_e': 0, 'c2_x': 0}
{'c1_a': 0, 'c1_h': 1, 'c2_e': 1, 'c2_x': 0}
```

```python
for key in sorted(oh.values.keys()):
    print(key)
    print(sorted(oh.values[key]))
```

```text
c1
['a', 'h']
c2
['e', 'x']
```

A subset of the features can be one-hot encoded by piping a [`compose.Select`](<https://riverml.xyz/dev/api/compose/Select>) into the `OneHotEncoder`.

```python
from river import compose

pp = compose.Select('c1') | preprocessing.OneHotEncoder()

for x in X:
    pp.learn_one(x)
    pprint(pp.transform_one(x))
```

```text
{'c1_u': 1}
{'c1_a': 1, 'c1_u': 0}
{'c1_a': 0, 'c1_i': 1, 'c1_u': 0}
{'c1_a': 0, 'c1_h': 1, 'c1_i': 0, 'c1_u': 0}
```

You can preserve the `c2` feature by using a union:

```python
pp = compose.Select('c1') | preprocessing.OneHotEncoder()
pp += compose.Select('c2')

for x in X:
    pp.learn_one(x)
    pprint(pp.transform_one(x))
```

```text
{'c1_u': 1, 'c2': 'd'}
{'c1_a': 1, 'c1_u': 0, 'c2': 'x'}
{'c1_a': 0, 'c1_i': 1, 'c1_u': 0, 'c2': 'h'}
{'c1_a': 0, 'c1_h': 1, 'c1_i': 0, 'c1_u': 0, 'c2': 'e'}
```

Similar to the above examples, we can also pass values as a list. This will one-hot encode all of the entries individually.

```python
X = [{'c1': ['u', 'a'], 'c2': ['d']},
    {'c1': ['a', 'b'], 'c2': ['x']},
    {'c1': ['i'], 'c2': ['h', 'z']},
    {'c1': ['h', 'b'], 'c2': ['e']}]

oh = preprocessing.OneHotEncoder(drop_zeros=True)
for x in X:
    oh.learn_one(x)
    pprint(oh.transform_one(x))
```

```text
{'c1_a': 1, 'c1_u': 1, 'c2_d': 1}
{'c1_a': 1, 'c1_b': 1, 'c2_x': 1}
{'c1_i': 1, 'c2_h': 1, 'c2_z': 1}
{'c1_b': 1, 'c1_h': 1, 'c2_e': 1}
```

Processing mini-batches is also possible.

```python
from pprint import pprint
import random
import string
import pandas as pd

random.seed(42)
alphabet = list(string.ascii_lowercase)
X = pd.DataFrame(
    {
        'c1': random.choice(alphabet),
        'c2': random.choice(alphabet),
    }
    for _ in range(3)
)
X
```

```text
  c1 c2
0  u  d
1  a  x
2  i  h
```

```python
oh = preprocessing.OneHotEncoder(drop_zeros=True)
df = oh.transform_many(X)
df.sort_index(axis="columns")
```

```text
   c1_a  c1_i  c1_u  c2_d  c2_h  c2_x
0     0     0     1     1     0     0
1     1     0     0     0     0     1
2     0     1     0     0     1     0
```

```python
oh = preprocessing.OneHotEncoder(drop_zeros=True, drop_first=True)
df = oh.transform_many(X)
df.sort_index(axis="columns")
```

```text
   c1_i  c1_u  c2_d  c2_h  c2_x
0     0     1     1     0     0
1     0     0     0     0     1
2     1     0     0     1     0
```

Here's an example where the zeros are kept:

```python
oh = preprocessing.OneHotEncoder(drop_zeros=False)
X_init = pd.DataFrame([{"c1": "Oranges", "c2": "Apples"}])
oh.learn_many(X_init)
oh.learn_many(X)

df = oh.transform_many(X)
df.sort_index(axis="columns")
```

```text
   c1_Oranges  c1_a  c1_i  c1_u  c2_Apples  c2_d  c2_h  c2_x
0           0     0     0     1          0     1     0     0
1           0     1     0     0          0     0     0     1
2           0     0     1     0          0     0     1     0
```

```python
df.dtypes.sort_index()
```

```text
c1_Oranges    Sparse[uint8, 0]
c1_a          Sparse[uint8, 0]
c1_i          Sparse[uint8, 0]
c1_u          Sparse[uint8, 0]
c2_Apples     Sparse[uint8, 0]
c2_d          Sparse[uint8, 0]
c2_h          Sparse[uint8, 0]
c2_x          Sparse[uint8, 0]
dtype: object
```

Explicit categories:

```python
oh = preprocessing.OneHotEncoder(categories=categories)

oh.learn_many(X)
df = oh.transform_many(X)
df.sort_index(axis="columns")
```

```text
   c1_a  c1_h  c2_e  c2_x
0     0     0     0     0
1     1     0     0     1
2     0     0     0     0
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
> One-hot encode a mini-batch of features.
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
> - **y** — defaults to `None`
>
> **Returns**
>
> `dict&#91;base.typing.FeatureName, Any&#93;`: The transformed values.
