# AdaGrad

AdaGrad optimizer.

## Parameters

- **lr** *(mutable)*

  *Type* → `int | float | optim.base.Scheduler`

  *Default* → `0.1`
- **eps**

  *Type* → `float`

  *Default* → `1e-08`

## Attributes

- **g2** (`collections.defaultdict`)

## Examples

```python
from river import datasets
from river import evaluate
from river import linear_model
from river import metrics
from river import optim
from river import preprocessing

dataset = datasets.Phishing()
optimizer = optim.AdaGrad()
model = (
    preprocessing.StandardScaler() |
    linear_model.LogisticRegression(optimizer)
)
metric = metrics.F1()

evaluate.progressive_val_score(dataset, model, metric)
```

```text
F1: 88.01%
```

## Methods

> [!NOTE]
>
> **look\_ahead**
>
> Updates a weight vector before a prediction is made.
>
> Parameters: w (dict): A dictionary of weight parameters. The weights are modified in-place. Returns: The updated weights.
>
> **Parameters**
>
> - **w** — `DictLike | VectorLike`

> [!NOTE]
>
> **step**
>
> Updates a weight vector given a gradient.
>
> **Parameters**
>
> - **w** — `DictLike | VectorLike`
> - **g** — `DictLike | VectorLike`
>
> **Returns**
>
> `DictLike | VectorLike`: The updated weights.

## References

- [Duchi, J., Hazan, E. and Singer, Y., 2011. Adaptive subgradient methods for online learning and stochastic optimization. Journal of machine learning research, 12(Jul), pp.2121-2159.](<http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf>)
