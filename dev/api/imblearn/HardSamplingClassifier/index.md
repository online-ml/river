# HardSamplingClassifier

Hard sampling classifier.

This wrapper enables a model to retrain on past samples who's output was hard to predict. This works by storing the hardest samples in a buffer of a fixed size. When a new sample arrives, the wrapped model is either trained on one of the buffered samples with a probability p or on the new sample with a probability (1 - p).

The hardness of an observation is evaluated with a loss function that compares the sample's ground truth with the wrapped model's prediction. If the buffer is not full, then the sample is added to the buffer. If the buffer is full and the new sample has a bigger loss than the lowest loss in the buffer, then the sample takes its place.

## Parameters

- **classifier**

  *Type* → `base.Classifier`
- **size**

  *Type* → `int`

  Size of the buffer.
- **p**

  *Type* → `float`

  Probability of updating the model with a sample from the buffer instead of a new incoming sample.
- **loss**

  *Type* → `optim.losses.BinaryLoss | optim.losses.MultiClassLoss | None`

  *Default* → `None`

  Criterion used to evaluate the hardness of a sample.
- **seed**

  *Type* → `int | None`

  *Default* → `None`

  Random seed.

## Attributes

- **classifier**

## Examples

```python
from river import datasets
from river import evaluate
from river import imblearn
from river import linear_model
from river import metrics
from river import optim
from river import preprocessing

model = (
    preprocessing.StandardScaler() |
    imblearn.HardSamplingClassifier(
        classifier=linear_model.LogisticRegression(),
        p=0.1,
        size=40,
        seed=42,
    )
)

evaluate.progressive_val_score(
    dataset=datasets.Phishing(),
    model=model,
    metric=metrics.ROCAUC(),
    print_every=500,
)
```

```text
[500] ROCAUC: 92.78%
[1,000] ROCAUC: 94.76%
[1,250] ROCAUC: 95.06%
ROCAUC: 95.06%
```

## Methods

> [!NOTE]
>
> **learn\_one**

> [!NOTE]
>
> **predict\_one**

> [!NOTE]
>
> **predict\_proba\_one**
>
> Predict the probability of each label for a dictionary of features `x`.
>
> **Parameters**
>
> - **x**
> - **kwargs**
>
> **Returns**
>
> A dictionary that associates a probability which each label.
