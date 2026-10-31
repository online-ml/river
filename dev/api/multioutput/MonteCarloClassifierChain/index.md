# MonteCarloClassifierChain

Monte Carlo Sampling Classifier Chains.

Probabilistic Classifier Chains using Monte Carlo sampling, as described in [^/dev/api/multioutput/MonteCarloClassifierChain/#1].

m samples are taken from the posterior distribution. Therefore we need a probabilistic interpretation of the output, and thus, this is a particular variety of ProbabilisticClassifierChain.

## Parameters

- **model**

  *Type* → `base.Classifier`
- **m**

  *Type* → `int`

  *Default* → `10`

  Number of samples to take from the posterior distribution.
- **seed**

  *Type* → `int | None`

  *Default* → `None`

  Random number generator seed for reproducibility.

## Examples

```python
from river import feature_selection
from river import linear_model
from river import metrics
from river import multioutput
from river import preprocessing
from river.datasets import synth

dataset = synth.Logical(seed=42, n_tiles=100)

model = multioutput.MonteCarloClassifierChain(
    model=linear_model.LogisticRegression(),
    m=10,
    seed=42
)

metric = metrics.multioutput.MicroAverage(metrics.Jaccard())

for x, y in dataset:
   y_pred = model.predict_one(x)
   y_pred = {k: y_pred.get(k, 0) for k in y}
   metric.update(y, y_pred)
   model.learn_one(x, y)

metric
```

```text
MicroAverage(Jaccard): 51.79%
```

## Methods

> [!NOTE]
>
> **learn\_one**
>
> Update the model with a set of features `x` and the labels `y`.
>
> **Parameters**
>
> - **x**
> - **y**
> - **kwargs**

> [!NOTE]
>
> **predict\_one**
>
> Predict the labels of a set of features `x`.
>
> **Parameters**
>
> - **x** — `dict[FeatureName, typing.Any]`
> - **kwargs** — `typing.Any`
>
> **Returns**
>
> `dict&#91;FeatureName, bool&#93;`: The predicted labels.

> [!NOTE]
>
> **predict\_proba\_one**
>
> Predict the probability of each label appearing given dictionary of features `x`.
>
> **Parameters**
>
> - **x**
> - **kwargs**
>
> **Returns**
>
> A dictionary that associates a probability which each label.

[^/dev/api/multioutput/MonteCarloClassifierChain/#1]: Read, J., Martino, L., &amp; Luengo, D. (2014). Efficient monte carlo methods for multi-dimensional learning with classifier chains. Pattern Recognition, 47(3), 1535-1546.
