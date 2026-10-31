# SGD

Plain stochastic gradient descent.

## Parameters

- **lr** *(mutable)*

  *Type* → `int | float | optim.base.Scheduler`

  *Default* → `0.01`

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
optimizer = optim.SGD(0.1)
model = (
    preprocessing.StandardScaler() |
    linear_model.LogisticRegression(optimizer)
)
metric = metrics.F1()

evaluate.progressive_val_score(dataset, model, metric)
```

```text
F1: 87.85%
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

- [Robbins, H. and Monro, S., 1951. A stochastic approximation method. The annals of mathematical statistics, pp.400-407](<https://pdfs.semanticscholar.org/34dd/d8865569c2c32dec9bf7ffc817ff42faaa01.pdf>)
