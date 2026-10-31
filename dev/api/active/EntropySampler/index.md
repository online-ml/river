# EntropySampler

Active learning classifier based on entropy measures.

The entropy sampler selects samples for labeling based on the entropy of the prediction. The higher the entropy, the more likely the sample will be selected for labeling. The entropy measure is normalized to \[0, 1\] and then raised to the power of the discount factor.

## Parameters

- **classifier**

  *Type* → `base.Classifier`

  The classifier to wrap.
- **discount\_factor**

  *Type* → `float`

  *Default* → `3`

  The discount factor to apply to the entropy measure. A value of 1 won't affect the entropy. The higher the discount factor, the more the entropy will be discounted, and the less likely samples will be selected for labeling. A value of 0 will select all samples for labeling. The discount factor is thus a way to control how many samples are selected for labeling.
- **seed**

  *Default* → `None`

  Random number generator seed for reproducibility.

## Examples

```python
from river import active
from river import datasets
from river import feature_extraction
from river import linear_model
from river import metrics

dataset = datasets.SMSSpam()
metric = metrics.Accuracy()
model = (
    feature_extraction.TFIDF(on='body') |
    linear_model.LogisticRegression()
)
model = active.EntropySampler(model, seed=42)

n_samples_used = 0
for x, y in dataset:
    y_pred, ask = model.predict_one(x)
    metric.update(y, y_pred)
    if ask:
        n_samples_used += 1
        model.learn_one(x, y)

metric
```

```text
Accuracy: 86.60%
```

```python
dataset.n_samples, n_samples_used
```

```text
(5574, 1921)
```

```python
print(f"{n_samples_used / dataset.n_samples:.2%}")
```

```text
34.46%
```

## Methods

> [!NOTE]
>
> **learn\_one**
>
> Update the model with a set of features `x` and a label `y`.
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
> Predict the label of `x` and indicate whether a label is needed.
>
> **Parameters**
>
> - **x**
> - **kwargs**
>
> **Returns**
>
> The predicted label.

> [!NOTE]
>
> **predict\_proba\_one**
>
> Predict the probability of each label for `x` and indicate whether a label is needed.
>
> **Parameters**
>
> - **x**
> - **kwargs**
>
> **Returns**
>
> A dictionary that associates a probability which each label.
