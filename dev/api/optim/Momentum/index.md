# Momentum

Momentum optimizer.

## Parameters

- **lr** *(mutable)*

  *Type* → `int | float | optim.base.Scheduler`

  *Default* → `0.1`
- **rho**

  *Type* → `float`

  *Default* → `0.9`

## Attributes

- **learning\_rate**

## Examples

```python
from river import datasets
from river import evaluate
from river import linear_model
from river import metrics
from river import optim
from river import preprocessing

dataset = datasets.Phishing()
optimizer = optim.Momentum()
model = (
    preprocessing.StandardScaler() |
    linear_model.LogisticRegression(optimizer)
)
metric = metrics.F1()

evaluate.progressive_val_score(dataset, model, metric)
```

```text
F1: 84.09%
```

## Methods

> [!NOTE]
>
> **look\_ahead**
>
> Update a weight vector before a prediction is made.
>
> **Parameters**
>
> - **w** — `DictLike | VectorLike`
>
> **Returns**
>
> `DictLike | VectorLike`: The updated weights.

> [!NOTE]
>
> **step**
>
> Update a weight vector given a gradient.
>
> **Parameters**
>
> - **w** — `DictLike | VectorLike`
> - **g** — `DictLike | VectorLike`
>
> **Returns**
>
> `DictLike | VectorLike`: The updated weights.
