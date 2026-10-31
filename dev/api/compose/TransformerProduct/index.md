# TransformerProduct

Computes interactions between the outputs of a set transformers.

This is for when you want to add interaction terms between groups of features. It may also be used an alternative to [`feature_extraction.PolynomialExtender`](<https://riverml.xyz/dev/api/feature-extraction/PolynomialExtender>) when the latter is overkill.

## Examples

Let's say we have a certain set of features with two groups. In practice these may be different namespaces, such one for items and the other for users.

```python
x = dict(
    a=0, b=1,  # group 1
    x=2, y=3   # group 2
)
```

We might want to add interaction terms between groups `('a', 'b')` and `('x', 'y')`, as so:

```python
from pprint import pprint
from river.compose import Select, TransformerProduct

product = TransformerProduct(
    Select('a', 'b'),
    Select('x', 'y')
)
pprint(product.transform_one(x))
```

```text
{'a*x': 0, 'a*y': 0, 'b*x': 2, 'b*y': 3}
```

This can also be done with the following shorthand:

```python
product = Select('a', 'b') * Select('x', 'y')
pprint(product.transform_one(x))
```

```text
{'a*x': 0, 'a*y': 0, 'b*x': 2, 'b*y': 3}
```

If you want to include the original terms, you can do something like this:

```python
group_1 = Select('a', 'b')
group_2 = Select('x', 'y')
product = group_1 + group_2 + group_1 * group_2
pprint(product.transform_one(x))
```

```text
{'a': 0, 'a*x': 0, 'a*y': 0, 'b': 1, 'b*x': 2, 'b*y': 3, 'x': 2, 'y': 3}
```

## Methods

> [!NOTE]
>
> **learn\_many**
>
> Update each transformer.
>
> **Parameters**
>
> - **X** — `IntoDataFrame`
> - **y** — `IntoSeries | None` — defaults to `None`

> [!NOTE]
>
> **learn\_one**
>
> Update each transformer.
>
> **Parameters**
>
> - **x** — `dict[base.typing.FeatureName, Any]`
> - **y** — defaults to `None`

> [!NOTE]
>
> **transform\_many**
>
> Passes the data through each transformer and packs the results together.
>
> The per-transformer outputs are concatenated column-wise via narwhals, so the result is rebuilt in the caller's own backend (pandas, polars, pyarrow, ...). All transformers see the same rows in the same order, so a positional horizontal concat matches the historical index-aligned `pandas.concat(axis=1)`.
>
> **Parameters**
>
> - **X** — `pd.DataFrame`

> [!NOTE]
>
> **transform\_one**
>
> Passes the data through each transformer and packs the results together.
>
> **Parameters**
>
> - **x** — `dict[base.typing.FeatureName, Any]`
