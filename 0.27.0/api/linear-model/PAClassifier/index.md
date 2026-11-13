# PAClassifier

Passive-aggressive learning for classification.

## Parameters

- **C**

  *Type* → `float`

  *Default* → `1.0`
- **mode**

  *Type* → `int`

  *Default* → `1`
- **learn\_intercept**

  *Type* → `bool`

  *Default* → `True`

## Examples

The following example is taken from [this blog post](<https://www.bonaccorso.eu/2017/10/06/ml-algorithms-addendum-passive-aggressive-algorithms/>).

```python
from river import linear_model
from river import metrics
from river import stream
import numpy as np
from sklearn import datasets
from sklearn import model_selection

np.random.seed(1000)
X, y = datasets.make_classification(
    n_samples=5000,
    n_features=4,
    n_informative=2,
    n_redundant=0,
    n_repeated=0,
    n_classes=2,
    n_clusters_per_class=2
)

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X,
    y,
    test_size=0.35,
    random_state=1000
)

model = linear_model.PAClassifier(
    C=0.01,
    mode=1
)

for xi, yi in stream.iter_array(X_train, y_train):
    model.learn_one(xi, yi)

metric = metrics.Accuracy() + metrics.LogLoss()

for xi, yi in stream.iter_array(X_test, y_test):
    metric.update(yi, model.predict_proba_one(xi))

print(metric)
```

```text
Accuracy: 88.46%
LogLoss: 0.325727...
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

## References

- [Crammer, K., Dekel, O., Keshet, J., Shalev-Shwartz, S. and Singer, Y., 2006. Online passive-aggressive algorithms. Journal of Machine Learning Research, 7(Mar), pp.551-585](<http://jmlr.csail.mit.edu/papers/volume7/crammer06a/crammer06a.pdf>)
