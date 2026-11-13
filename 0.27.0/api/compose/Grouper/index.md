# Grouper

Applies a transformer within different groups.

This transformer allows you to split your data into groups and apply a transformer within each group. This happens in a streaming manner, which means that the groups are discovered online. A separate copy of the provided transformer is made whenever a new group appears. The groups are defined according to one or more keys.

## Parameters

- **transformer**

  *Type* → `base.BaseTransformer`
- **by**

  *Type* → `base.typing.FeatureName | list[base.typing.FeatureName]`

  The field on which to group the data. This can either by a single value, or a list of values.

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
