# AMSGrad

AMSGrad optimizer.

## Parameters

- **lr** *(mutable)*

  *Type* → `int | float | optim.base.Scheduler`

  *Default* → `0.1`

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
- **correct\_bias**

  *Type* → `bool`

  *Default* → `True`

## Attributes

- **m** (`collections.defaultdict`)
- **v** (`collections.defaultdict`)
- **v\_hat** (`collections.defaultdict`)

## Examples

```python
from river import datasets
from river import evaluate
from river import linear_model
from river import metrics
from river import optim
from river import preprocessing

dataset = datasets.Phishing()
optimizer = optim.AMSGrad()
model = (
    preprocessing.StandardScaler() |
    linear_model.LogisticRegression(optimizer)
)
metric = metrics.F1()

evaluate.progressive_val_score(dataset, model, metric)
```

```text
F1: 86.60%
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

- [Reddi, S.J., Kale, S. and Kumar, S., 2018. On the convergence of Adam and beyond. In Sixth International Conference on Learning Representations (ICLR 2018)](<https://arxiv.org/pdf/1904.09237.pdf>)
