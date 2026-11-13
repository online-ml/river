# PoissonInclusion

Randomly selects features with an inclusion trial.

When a new feature is encountered, it is selected with probability `p`. The number of times a feature needs to beseen before it is added to the model follows a geometric distribution with expected value `1 / p`. This feature selection method is meant to be used when you have a very large amount of sparse features.

## Parameters

- **p**

  *Type* → `float`

  Probability of including a feature the first time it is encountered.
- **seed**

  *Type* → `int | None`

  *Default* → `None`

  Random seed value used for reproducibility.

## Examples

```python
from river import datasets
from river import feature_selection
from river import stream

selector = feature_selection.PoissonInclusion(p=0.1, seed=42)

dataset = iter(datasets.TrumpApproval())

feature_names = next(dataset)[0].keys()
n = 0

while True:
    x, y = next(dataset)
    xt = selector.transform_one(x)
    if xt.keys() == feature_names:
        break
    n += 1

n
```

```text
12
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
> - **x** — `dict[base.typing.FeatureName, typing.Any]`
>
> **Returns**
>
> `dict&#91;base.typing.FeatureName, typing.Any&#93;`: The transformed values.

## References

- [McMahan, H.B., Holt, G., Sculley, D., Young, M., Ebner, D., Grady, J., Nie, L., Phillips, T., Davydov, E., Golovin, D. and Chikkerur, S., 2013, August. Ad click prediction: a view from the trenches. In Proceedings of the 19th ACM SIGKDD international conference on Knowledge discovery and data mining (pp. 1222-1230)](<https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/41159.pdf>)
