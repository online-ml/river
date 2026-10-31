# Renamer

Renames features following substitution rules.

## Parameters

- **mapping**

  *Type* → `dict[str, str]`

  Dictionary describing substitution rules. Keys in `mapping` that are not a feature's name are silently ignored.

## Examples

```python
from river import compose

mapping = {'a': 'v', 'c': 'o'}
x = {'a': 42, 'b': 12}
compose.Renamer(mapping).transform_one(x)
```

```text
{'b': 12, 'v': 42}
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
