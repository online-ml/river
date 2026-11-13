# Discard

Removes features.

This can be used in a pipeline when you want to remove certain features. The `transform_one` method is pure, and therefore returns a fresh new dictionary instead of removing the specified keys from the input.

## Examples

```python
from river import compose

x = {'a': 42, 'b': 12, 'c': 13}
compose.Discard('a', 'b').transform_one(x)
```

```text
{'c': 13}
```

You can chain a discarder with any estimator in order to apply said estimator to the desired features.

```python
from river import feature_extraction as fx

x = {'sales': 10, 'shop': 'Ikea', 'country': 'Sweden'}

pipeline = (
    compose.Discard('shop', 'country') |
    fx.PolynomialExtender()
)
pipeline.transform_one(x)
```

```text
{'sales': 10, 'sales*sales': 100}
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
> - **x** — `dict`
>
> **Returns**
>
> `dict`: The transformed values.
