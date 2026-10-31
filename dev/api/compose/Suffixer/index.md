# Suffixer

Appends a suffix on features names.

## Parameters

- **suffix**

  *Type* → `str`

## Examples

```python
from river import compose

x = {'a': 42, 'b': 12}
compose.Suffixer('_suffix').transform_one(x)
```

```text
{'a_suffix': 42, 'b_suffix': 12}
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
