# AdaBound

AdaBound optimizer.

## Parameters

- **lr** *(mutable)*

  *Type* → `int | float | optim.base.Scheduler`

  *Default* → `0.001`

  The learning rate.
- **beta\_1**

  *Type* → `float`

  *Default* → `0.9`
- **beta\_2**

  *Type* → `float`

  *Default* → `0.999`
- **eps**

  *Type* → `float`

  *Default* → `1e-08`
- **gamma**

  *Type* → `float`

  *Default* → `0.001`
- **final\_lr**

  *Type* → `float`

  *Default* → `0.1`

## Attributes

- **m** (`collections.defaultdict`)
- **s** (`collections.defaultdict`)

## Examples

```python
from river import datasets
from river import evaluate
from river import linear_model
from river import metrics
from river import optim
from river import preprocessing

dataset = datasets.Phishing()
optimizer = optim.AdaBound()
model = (
    preprocessing.StandardScaler() |
    linear_model.LogisticRegression(optimizer)
)
metric = metrics.F1()

evaluate.progressive_val_score(dataset, model, metric)
```

```text
F1: 88.06%
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

## References

- [Luo, L., Xiong, Y., Liu, Y. and Sun, X., 2019. Adaptive gradient methods with dynamic bound of learning rate. In Seventh International Conference on Learning Representations (ICLR 2019)](<https://arxiv.org/abs/1902.09843>)
