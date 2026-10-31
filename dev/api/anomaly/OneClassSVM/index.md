# OneClassSVM

One-class SVM for anomaly detection.

This is a stochastic implementation of the one-class SVM algorithm, and will not exactly match its batch formulation.

It is encouraged to scale the data upstream with [`preprocessing.StandardScaler`](<https://riverml.xyz/dev/api/preprocessing/StandardScaler>), as well as use [`feature_extraction.RBFSampler`](<https://riverml.xyz/dev/api/feature-extraction/RBFSampler>) to capture non-linearities.

## Parameters

- **nu**

  *Default* → `0.1`

  An upper bound on the fraction of training errors and a lower bound of the fraction of support vectors. You can think of it as the expected fraction of anomalies.
- **optimizer** *(mutable)*

  *Type* → `optim.base.Optimizer | None`

  *Default* → `None`

  The sequential optimizer used for updating the weights.
- **intercept\_lr** *(mutable)*

  *Type* → `optim.base.Scheduler | float`

  *Default* → `0.01`

  Learning rate scheduler used for updating the intercept. A `optim.schedulers.Constant` is used if a `float` is provided. The intercept is not updated when this is set to 0.
- **clip\_gradient** *(mutable)*

  *Default* → `1000000000000.0`

  Clips the absolute value of each gradient value.
- **initializer** *(mutable)*

  *Type* → `optim.base.Initializer | None`

  *Default* → `None`

  Weights initialization scheme.

## Attributes

- **weights**

## Examples

```python
from river import anomaly
from river import compose
from river import datasets
from river import metrics
from river import preprocessing

model = anomaly.QuantileFilter(
    anomaly.OneClassSVM(nu=0.2),
    q=0.995
)

auc = metrics.ROCAUC()

for x, y in datasets.CreditCard().take(2500):
    score = model.score_one(x)
    is_anomaly = model.classify(score)
    model.learn_one(x)
    auc.update(y, is_anomaly)

auc
```

```text
ROCAUC: 74.68%
```

You can also use the [`evaluate.progressive_val_score`](<https://riverml.xyz/dev/api/evaluate/progressive-val-score>) function to evaluate the model on a data stream.

```python
from river import evaluate

model = model.clone()

evaluate.progressive_val_score(
    dataset=datasets.CreditCard().take(2500),
    model=model,
    metric=metrics.ROCAUC(),
    print_every=1000
)
```

```text
[1,000] ROCAUC: 74.40%
[2,000] ROCAUC: 74.60%
[2,500] ROCAUC: 74.68%
ROCAUC: 74.68%
```

## Methods

> [!NOTE]
>
> **learn\_many**

> [!NOTE]
>
> **learn\_one**
>
> Update the model.
>
> **Parameters**
>
> - **x** — `dict[base.typing.FeatureName, typing.Any]`

> [!NOTE]
>
> **score\_one**
>
> Return an outlier score.
>
> A high score is indicative of an anomaly. A low score corresponds to a normal observation.
>
> **Parameters**
>
> - **x**
>
> **Returns**
>
> An anomaly score. A high score is indicative of an anomaly. A low score corresponds a
