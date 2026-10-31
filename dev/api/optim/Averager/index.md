# Averager

Averaged stochastic gradient descent.

This is a wrapper that can be applied to any stochastic gradient descent optimiser. Note that this implementation differs than what may be found elsewhere. Essentially, the average of the weights is usually only used at the end of the optimisation, once all the data has been seen. However, in this implementation the optimiser returns the current averaged weights.

## Parameters

- **optimizer**

  *Type* → `optim.base.Optimizer`

  An optimizer for which the produced weights will be averaged.
- **start**

  *Type* → `int`

  *Default* → `0`

  Indicates the number of iterations to wait before starting the average. Essentially, nothing happens differently before the number of iterations reaches this value.

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
optimizer = optim.Averager(optim.SGD(0.01), 100)
model = (
    preprocessing.StandardScaler() |
    linear_model.LogisticRegression(optimizer)
)
metric = metrics.F1()

evaluate.progressive_val_score(dataset, model, metric)
```

```text
F1: 87.97%
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

- [Bottou, L., 2010. Large-scale machine learning with stochastic gradient descent. In Proceedings of COMPSTAT'2010 (pp. 177-186). Physica-Verlag HD.](<https://leon.bottou.org/publications/pdf/compstat-2010.pdf>)
- [Stochastic Algorithms for One-Pass Learning slides by Léon Bottou](<https://leon.bottou.org/slides/onepass/onepass.pdf>)
- [Xu, W., 2011. Towards optimal one pass large scale learning with averaged stochastic gradient descent. arXiv preprint arXiv:1107.2490.](<https://arxiv.org/pdf/1107.2490.pdf>)
