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

- [Reddi, S.J., Kale, S. and Kumar, S., 2019. On the convergence of adam and beyond. arXiv preprint arXiv:1904.09237](<https://arxiv.org/pdf/1904.09237.pdf>)
