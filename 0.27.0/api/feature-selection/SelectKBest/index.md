# SelectKBest

Removes all but the \\(k\\) highest scoring features.

## Parameters

- **similarity**

  *Type* → `stats.base.Bivariate`
- **k**

  *Type* → `int`

  *Default* → `10`

  The number of features to keep.
- **use\_abs**

  *Type* → `bool`

  *Default* → `False`

  A boolean indicating whether to rank features based on the absolute value of their scores. This is particularly useful when the similarity metric can produce negative values, such as Pearson correlation. Defaults to `False`.

## Attributes

- **similarities** (`dict`)

  The similarity instances used for each feature.
- **leaderboard** (`dict`)

  The actual similarity measures.

## Examples

```python
from pprint import pprint
from river import feature_selection
from river import stats
from river import stream
from sklearn import datasets

X, y = datasets.make_regression(
    n_samples=100,
    n_features=10,
    n_informative=2,
    random_state=42
)

selector = feature_selection.SelectKBest(
    similarity=stats.PearsonCorr(),
    k=2
)

for xi, yi, in stream.iter_array(X, y):
    selector.learn_one(xi, yi)

pprint(selector.leaderboard)
```

```text
Counter({9: 0.7898,
        7: 0.5444,
        8: 0.1062,
        2: 0.0638,
        4: 0.0538,
        5: 0.0271,
        1: -0.0312,
        6: -0.0657,
        3: -0.1501,
        0: -0.1895})
```

```python
selector.transform_one(xi)
```

```text
{7: -1.2795, 9: -1.8408}
```

```python
import random

random.seed(42)
X_abs = [[random.random() for _ in range(3)] for _ in range(100)]
y_abs = [
    0.6 * x[0]
    - 0.9 * x[1]
    + 0.1 * x[2]
    + random.gauss(0, 0.1)
    for x in X_abs
]

selector_no_abs = feature_selection.SelectKBest(
    stats.PearsonCorr(),
    k=1,
    use_abs=False
)
for xi, yi in stream.iter_array(X_abs, y_abs):
    selector_no_abs.learn_one(xi, yi)
pprint(selector_no_abs.leaderboard)
```

```text
Counter({0: 0.5683236302249015,
         2: -0.09937590098236333,
         1: -0.7655616041162767})
```

```python
selector_no_abs.transform_one({i: v for i, v in enumerate(X_abs[-1])})
```

```text
{0: 0.009669699608339966}
```

```python
selector_with_abs = feature_selection.SelectKBest(
    stats.PearsonCorr(),
    k=1,
    use_abs=True
)
for xi, yi in stream.iter_array(X_abs, y_abs):
    selector_with_abs.learn_one(xi, yi)
selector_with_abs.transform_one({i: v for i, v in enumerate(X_abs[-1])})
```

```text
{1: 0.07524386007376704}
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
> - **x** — `dict[base.typing.FeatureName, typing.Any]`
> - **y** — `base.typing.Target`

> [!NOTE]
>
> **transform\_one**
>
> Transform a set of features `x`.
>
> **Parameters**
>
> - **x** — `dict[base.typing.FeatureName, typing.Any]`
>
> **Returns**
>
> `dict&#91;base.typing.FeatureName, typing.Any&#93;`: The transformed values.
