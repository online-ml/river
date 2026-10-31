# AdPredictor

Bayesian online probit regression for click-through-rate prediction.

AdPredictor, used at some point by Microsoft for CTR prediction in Bing's sponsored search [^/dev/api/linear-model/AdPredictor/#1], keeps a Gaussian belief over each feature weight rather than a point estimate. It predicts via a probit link and learns in a single pass, scaling each weight's step size by its own uncertainty. It shines on the sparse, high-cardinality, categorical data of ad logs: it yields well-calibrated probabilities and exposes uncertainty for exploration. Like the other linear models, its cost per example scales only with the number of active features. Plain logistic regression is simpler and just as good on dense, low-dimensional numeric data.

Features are expected to be a sparse active set: a key in `x` is active when its value is truthy (its magnitude is otherwise ignored), so one-hot encode or bucket inputs. Use [`preprocessing.OneHotEncoder`](<https://riverml.xyz/dev/api/preprocessing/OneHotEncoder>)(drop\_zeros=True)\` to feed only the present categories.

## Parameters

- **beta**

  *Type* → `float`

  *Default* → `0.1`

  Standard deviation of the per-example label noise; the prediction step size shrinks as `beta` grows.
- **prior\_probability**

  *Type* → `float`

  *Default* → `0.5`

  Base-rate CTR used to initialise the bias weight, so the model predicts this value before seeing any data. The bias is calibrated lazily from the number of active features in the first observed example.
- **epsilon**

  *Type* → `float`

  *Default* → `0.05`

  Variance-dynamics rate in \[0, 1). Each update nudges weights a fraction `epsilon` of the way back toward the unit prior, preventing variances from collapsing and letting the model track drift. Set to 0 to disable.

## Examples

```python
from river import linear_model

model = linear_model.AdPredictor(beta=0.1, prior_probability=0.5)
```

With no data seen yet, the model predicts the base rate:

```python
model.predict_proba_one({"a": 1, "b": 1})[True]
```

```text
0.5
```

After repeatedly seeing the same clicked impression, its belief moves up:

```python
for _ in range(50):
    model.learn_one({"a": 1, "b": 1}, True)
model.predict_proba_one({"a": 1, "b": 1})[True] > 0.9
```

```text
True
```

On a real ad-click stream, one-hot encode the fields into a sparse active set and evaluate with progressive validation:

```python
from river import datasets, metrics, preprocessing

dataset = datasets.CriteoAds()
model = preprocessing.OneHotEncoder(drop_zeros=True) | linear_model.AdPredictor()
metric = metrics.ROCAUC()

for x, y in dataset.take(10_000):
    metric.update(y, model.predict_proba_one(x))
    model.learn_one(x, y)
metric
```

```text
ROCAUC: 64.95%
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
> - **x** — `dict[base.typing.FeatureName, typing.Any]`
> - **y** — `base.typing.ClfTarget`

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
> **predict\_proba\_one**
>
> Predict the probability of each label for a dictionary of features `x`.
>
> **Parameters**
>
> - **x** — `dict[base.typing.FeatureName, typing.Any]`
> - **kwargs** — `typing.Any`
>
> **Returns**
>
> `dict&#91;base.typing.ClfTarget, float&#93;`: A dictionary that associates a probability which each label.

[^/dev/api/linear-model/AdPredictor/#1]: Graepel, T., Candela, J.Q., Borchert, T. and Herbrich, R., 2010. [Web-scale Bayesian click-through rate prediction for sponsored search advertising in Microsoft's Bing search engine](<https://quinonero.net/Publications/AdPredictorICML2010-final.pdf>). ICML 2010.
