# GaussianNB

Gaussian Naive Bayes.

A Gaussian distribution \\(G\_{cf}\\) is maintained for each class \\(c\\) and each feature \\(f\\). Each Gaussian is updated using the amount associated with each feature; the details can be be found in [`proba.Gaussian`](<https://riverml.xyz/dev/api/proba/Gaussian>). The joint log-likelihood is then obtained by summing the log probabilities of each feature associated with each class.

## Examples

```python
from river import naive_bayes
from river import stream
import numpy as np

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
Y = np.array([1, 1, 1, 2, 2, 2])

model = naive_bayes.GaussianNB()

for x, y in stream.iter_array(X, Y):
    model.learn_one(x, y)

model.predict_one({0: -0.8, 1: -1})
```

```text
1
```

You can also train the model and make predictions in mini-batch mode.

```python
import pandas as pd

model = naive_bayes.GaussianNB()
model.learn_many(pd.DataFrame(X), pd.Series(Y))
model.predict_many(pd.DataFrame([[-0.8, -1], [2.8, 1.5]]))
```

```text
0    1
1    2
dtype: int64
```

## Methods

> [!NOTE]
>
> **joint\_log\_likelihood**
>
> Compute the unnormalized posterior log-likelihood of x.
>
> The log-likelihood is `log P(c) + log P(x|c)`.
>
> **Parameters**
>
> - **x** — `dict`

> [!NOTE]
>
> **joint\_log\_likelihood\_many**
>
> Compute the joint log-likelihoods for a batch of feature vectors.
>
> **Parameters**
>
> - **X** — `IntoDataFrame`
>
> **Returns**
>
> `IntoDataFrame`: Input samples joint log-likelihoods.

> [!NOTE]
>
> **learn\_many**
>
> Learn from a batch of feature vectors.
>
> **Parameters**
>
> - **X** — `IntoDataFrame`
> - **y** — `IntoSeries`

> [!NOTE]
>
> **learn\_one**
>
> Update the model with a set of features `x` and a label `y`.
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
