# BayesianLinearRegression

Bayesian linear regression.

An advantage of Bayesian linear regression over standard linear regression is that features do not have to scaled beforehand. Another attractive property is that this flavor of linear regression is somewhat insensitive to its hyperparameters. Finally, this model can output instead a predictive distribution rather than just a point estimate.

The downside is that the learning step runs in `O(n^2)` time, whereas the learning step of standard linear regression takes `O(n)` time.

## Parameters

- **alpha**

  *Type* → `float`

  *Default* → `1`

  Prior parameter.
- **beta**

  *Type* → `float`

  *Default* → `1`

  Noise parameter.
- **smoothing**

  *Type* → `float | None`

  *Default* → `None`

  Smoothing allows the model to gradually "forget" the past, and focus on the more recent data. It thus enables the model to deal with concept drift. Due to the current implementation, activating smoothing may slow down the model.

## Examples

```python
from river import datasets
from river import evaluate
from river import linear_model
from river import metrics

dataset = datasets.TrumpApproval()
model = linear_model.BayesianLinearRegression()
metric = metrics.MAE()

evaluate.progressive_val_score(dataset, model, metric)
```

```text
MAE: 0.586...
```

```python
x, _ = next(iter(dataset))
model.predict_one(x)
```

```text
43.855...
```

```python
model.predict_dist_one(x)
```

```text
𝒩(μ=43.85..., σ=1.00...)
```

The `smoothing` parameter can be set to make the model robust to drift. The parameter is expected to be between 0 and 1. To exemplify, let's generate some simulation data with an abrupt concept drift right in the middle.

```python
import itertools
import random

def random_data(coefs, n, seed=42):
    rng = random.Random(seed)
    for _ in range(n):
        x = {i: rng.random() for i, c in enumerate(coefs)}
        y = sum(c * xi for c, xi in zip(coefs, x.values()))
        yield x, y
```

Here's how the model performs without any smoothing:

```python
model = linear_model.BayesianLinearRegression()
dataset = itertools.chain(
    random_data([0.1, 3], 100),
    random_data([10, -2], 100)
)
metric = metrics.MAE()
evaluate.progressive_val_score(dataset, model, metric)
```

```text
MAE: 1.284...
```

And here's how it performs with some smoothing:

```python
model = linear_model.BayesianLinearRegression(smoothing=0.8)
dataset = itertools.chain(
    random_data([0.1, 3], 100),
    random_data([10, -2], 100)
)
metric = metrics.MAE()
evaluate.progressive_val_score(dataset, model, metric)
```

```text
MAE: 0.159...
```

Smoothing allows the model to gradually "forget" the past, and focus on the more recent data.

Note how this works better than standard linear regression, even when using an aggressive learning rate.

```python
from river import optim
model = linear_model.LinearRegression(optimizer=optim.SGD(0.5))
dataset = itertools.chain(
    random_data([0.1, 3], 100),
    random_data([10, -2], 100)
)
metric = metrics.MAE()
evaluate.progressive_val_score(dataset, model, metric)
```

```text
MAE: 0.242...
```

Features that are absent from `x` in `learn_one` are treated as observed values of 0. This is the right default when features are centered around zero (e.g. sparse counts or indicators), but biases the posterior when typical feature values are far from zero. In that case, combine the model with [`preprocessing.StatImputer`](<https://riverml.xyz/dev/api/preprocessing/StatImputer>) in a pipeline so that missing observations are filled with a running statistic before they reach the model:

```python
from river import preprocessing
from river import stats

def with_missing(dataset, p_missing=0.2, seed=42):
    rng = random.Random(seed)
    for x, y in dataset:
        x = {f: (None if rng.random() < p_missing else v) for f, v in x.items()}
        yield x, y

features = ['ordinal_date', 'gallup', 'ipsos', 'morning_consult',
            'rasmussen', 'you_gov']
model = (
    preprocessing.StatImputer(*[(f, stats.Mean()) for f in features])
    | linear_model.BayesianLinearRegression()
)
metric = metrics.MAE()
evaluate.progressive_val_score(
    with_missing(datasets.TrumpApproval()), model, metric
)
```

```text
MAE: 0.774...
```

## Methods

> [!NOTE]
>
> **learn\_many**
>
> Update the model with a mini-batch of features `X` and real-valued targets `y`.
>
> **Parameters**
>
> - **X** — `IntoDataFrame`
> - **y** — `IntoSeries`

> [!NOTE]
>
> **learn\_one**
>
> Fits to a set of features `x` and a real-valued target `y`.
>
> **Parameters**
>
> - **x** — `dict[base.typing.FeatureName, typing.Any]`
> - **y** — `base.typing.RegTarget`

> [!NOTE]
>
> **predict\_dist\_one**

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
> `IntoSeries`: The predicted outcomes.

> [!NOTE]
>
> **predict\_one**
>
> Predict the output of features `x`.
>
> **Parameters**
>
> - **x** — `dict[base.typing.FeatureName, typing.Any]`
>
> **Returns**
>
> `base.typing.RegTarget`: The prediction.

## References

- [Pattern Recognition and Machine Learning, page 52 — Christopher M. Bishop](<https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf>)
- [Bayesian/Streaming Algorithms — Vincent Warmerdam](<https://koaning.io/posts/bayesian-propto-streaming/>)
- [Bayesian linear regression for practitioners — Max Halford](<https://maxhalford.github.io/blog/bayesian-linear-regression/>)
