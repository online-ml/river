# Binarizer

Binarize the data to 0 or 1 according to a threshold.

## Parameters

- **threshold**

  *Default* → `0.0`

  Values above this are replaced by 1 and the others by 0.
- **dtype**

  *Default* → `<class 'bool'>`

  The desired data type to apply.

## Examples

```python
import river
import numpy as np

rng = np.random.RandomState(42)
X = [{'x1': v, 'x2': int(v)} for v in rng.uniform(low=-4, high=4, size=6)]

binarizer = river.preprocessing.Binarizer()
for x in X:
    binarizer.learn_one(x)
    print(binarizer.transform_one(x))
```

```text
{'x1': False, 'x2': False}
{'x1': True, 'x2': True}
{'x1': True, 'x2': True}
{'x1': True, 'x2': False}
{'x1': False, 'x2': False}
{'x1': False, 'x2': False}
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
