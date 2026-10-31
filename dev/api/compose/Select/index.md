# Select

Selects features.

This can be used in a pipeline when you want to select certain features. The `transform_one` method is pure, and therefore returns a fresh new dictionary instead of filtering the specified keys from the input.

## Examples

```python
from river import compose

x = {'a': 42, 'b': 12, 'c': 13}
compose.Select('c').transform_one(x)
```

```text
{'c': 13}
```

You can chain a selector with any estimator in order to apply said estimator to the desired features.

```python
from river import feature_extraction as fx

x = {'sales': 10, 'shop': 'Ikea', 'country': 'Sweden'}

pipeline = (
    compose.Select('sales') |
    fx.PolynomialExtender()
)
pipeline.transform_one(x)
```

```text
{'sales': 10, 'sales*sales': 100}
```

This transformer also supports mini-batch processing:

```python
import random
from river import compose

random.seed(42)
X = [{"x_1": random.uniform(8, 12), "x_2": random.uniform(8, 12)} for _ in range(6)]
for x in X:
    print(x)
```

```text
{'x_1': 10.557707193831535, 'x_2': 8.100043020890668}
{'x_1': 9.100117273476478, 'x_2': 8.892842952595291}
{'x_1': 10.94588485665605, 'x_2': 10.706797949691644}
{'x_1': 11.568718270819382, 'x_2': 8.347755330517664}
{'x_1': 9.687687278741082, 'x_2': 8.119188877752281}
{'x_1': 8.874551899214413, 'x_2': 10.021421152413449}
```

```python
import pandas as pd
X = pd.DataFrame.from_dict(X)
```

You can then call `transform_many` to transform a mini-batch of features:

```python
compose.Select('x_2').transform_many(X)
```

```text
    x_2
0   8.100043
1   8.892843
2  10.706798
3   8.347755
4   8.119189
5  10.021421
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
> - **X** — `pd.DataFrame`

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
> Transform a mini-batch of features.
>
> **Parameters**
>
> - **X** — `IntoDataFrameT`
>
> **Returns**
>
> `IntoDataFrameT`: A new DataFrame.

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
