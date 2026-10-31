# CategoricalNB

Naive Bayes classifier for categorical features.

Categorical Naive Bayes learns a separate model for each categorical feature. For each class, a frequency is maintained for every value seen for every feature. Prediction is done by computing the joint log-likelihood of each class given the observed category values.

## Parameters

- **alpha**

  *Default* → `1.0`

  Additive (Laplace/Lidstone) smoothing parameter (use 0 for no smoothing).

## Attributes

- **class\_counts** (`collections.Counter`)

  Number of times each class has been seen.
- **feature\_counts** (`collections.defaultdict`)

  Total frequencies per feature, value, and class.

## Examples

```python
from river import naive_bayes

X = [
    {"outlook": "sunny", "temp": "hot"},
    {"outlook": "sunny", "temp": "hot"},
    {"outlook": "sunny", "temp": "mild"},
    {"outlook": "sunny", "temp": "cool"},
    {"outlook": "rainy", "temp": "cool"},
    {"outlook": "rainy", "temp": "cool"},
    {"outlook": "rainy", "temp": "mild"},
    {"outlook": "rainy", "temp": "hot"},
    {"outlook": "overcast", "temp": "cool"},
    {"outlook": "overcast", "temp": "mild"},
    {"outlook": "overcast", "temp": "mild"},
    {"outlook": "overcast", "temp": "hot"},
]

y = ["no", "no", "no", "no", "yes", "yes", "yes", "yes",
     "yes", "yes", "yes", "yes"]

model = naive_bayes.CategoricalNB(alpha=1)

for x, yi in zip(X, y):
    model.learn_one(x, yi)

model.predict_one({"outlook": "sunny", "temp": "mild"})
```

```text
'no'
```

```python
model.predict_proba_one({"outlook": "sunny", "temp": "mild"})
```

```text
{'no': 0.755..., 'yes': 0.244...}
```

You can also train and predict in mini-batch mode.

```python
import pandas as pd

df = pd.DataFrame(X)
y = pd.Series(y)

batch_model = naive_bayes.CategoricalNB(alpha=1)
batch_model.learn_many(df, y)

unseen = pd.DataFrame([{"outlook": "rainy", "temp": "cool"}])

batch_model.predict_many(unseen).tolist()
```

```text
['yes']
```

```python
batch_model.predict_proba_many(unseen)
```

```text
           no       yes
0  0.109900  0.890100
```

## Methods

> [!NOTE]
>
> **joint\_log\_likelihood**
>
> Computes the joint log likelihood of input features.
>
> **Parameters**
>
> - **x** — `dict`
>
> **Returns**
>
> `float`: Mapping between classes and joint log likelihood.

> [!NOTE]
>
> **joint\_log\_likelihood\_many**
>
> Computes the joint log likelihood of input features.
>
> **Parameters**
>
> - **X** — `IntoDataFrame`
>
> **Returns**
>
> `IntoDataFrame`: Input samples joint log likelihood.

> [!NOTE]
>
> **learn\_many**
>
> Learn from a batch of categorical feature vectors.
>
> **Parameters**
>
> - **X** — `IntoDataFrame`
> - **y** — `IntoSeries`

> [!NOTE]
>
> **learn\_one**
>
> Updates the model with a single observation.
>
> **Parameters**
>
> - **x** — `dict[base.typing.FeatureName, Any]`
> - **y** — `base.typing.ClfTarget`

> [!NOTE]
>
> **p\_class**

> [!NOTE]
>
> **p\_class\_many**

> [!NOTE]
>
> **p\_feature\_given\_class**

> [!NOTE]
>
> **predict\_many**
>
> Predict the outcome for each given sample.
>
> **Parameters**
>
> - **X** — `IntoDataFrame`
>
> **Returns**
>
> `pd.Series`: The predicted labels.

> [!NOTE]
>
> **predict\_one**
>
> Predict the label of a set of features `x`.
>
> **Parameters**
>
> - **x** — `dict[base.typing.FeatureName, Any]`
> - **kwargs** — `Any`
>
> **Returns**
>
> `base.typing.ClfTarget | None`: The predicted label.

> [!NOTE]
>
> **predict\_proba\_many**
>
> Return probabilities using the log-likelihoods in mini-batchs setting.
>
> **Parameters**
>
> - **X** — `IntoDataFrame`

> [!NOTE]
>
> **predict\_proba\_one**
>
> Return probabilities using the log-likelihoods.
>
> **Parameters**
>
> - **x** — `dict[base.typing.FeatureName, Any]`

## References

- [CategoricalNB](<https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.CategoricalNB.html>)
