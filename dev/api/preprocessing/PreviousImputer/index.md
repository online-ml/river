# PreviousImputer

Imputes missing values by using the most recent value.

## Examples

```python
from river import preprocessing

imputer = preprocessing.PreviousImputer()

imputer.learn_one({'x': 1, 'y': 2})
imputer.transform_one({'y': None})
```

```text
{'y': 2}
```

```python
imputer.transform_one({'x': None})
```

```text
{'x': 1}
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
