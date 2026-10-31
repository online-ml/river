# FTRLProximal

FTRL-Proximal optimizer.

## Parameters

- **alpha**

  *Type* → `float`

  *Default* → `0.05`
- **beta**

  *Type* → `float`

  *Default* → `1.0`
- **l1**

  *Type* → `float`

  *Default* → `0.0`
- **l2**

  *Type* → `float`

  *Default* → `1.0`

## Attributes

- **z** (`collections.defaultdict`)
- **n** (`collections.defaultdict`)

## Examples

```python
from river import datasets
from river import evaluate
from river import linear_model
from river import metrics
from river import optim
from river import preprocessing

dataset = datasets.Phishing()
optimizer = optim.FTRLProximal()
model = (
    preprocessing.StandardScaler() |
    linear_model.LogisticRegression(optimizer)
)
metric = metrics.F1()

evaluate.progressive_val_score(dataset, model, metric)
```

```text
F1: 87.56%
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

- [McMahan, H.B., Holt, G., Sculley, D., Young, M., Ebner, D., Grady, J., Nie, L., Phillips, T., Davydov, E., Golovin, D. and Chikkerur, S., 2013, August. Ad click prediction: a view from the trenches. In Proceedings of the 19th ACM SIGKDD international conference on Knowledge discovery and data mining (pp. 1222-1230)](<https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/41159.pdf>)
- [Tensorflow's `FtrlOptimizer`](<https://www.tensorflow.org/api_docs/python/tf/train/FtrlOptimizer>)
