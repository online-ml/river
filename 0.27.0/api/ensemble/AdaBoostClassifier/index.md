# AdaBoostClassifier

Boosting for classification.

For each incoming observation, each model's `learn_one` method is called `k` times where `k` is sampled from a Poisson distribution of parameter lambda. The lambda parameter is updated when the weaks learners fit successively the same observation.

## Parameters

- **model**

  *Type* → `base.Classifier`

  The classifier to boost.
- **n\_models**

  *Default* → `10`

  The number of models in the ensemble.
- **seed**

  *Type* → `int | None`

  *Default* → `None`

  Random number generator seed for reproducibility.

## Attributes

- **models**

## Examples

In the following example three tree classifiers are boosted together. The performance is slightly better than when using a single tree.

```python
from river import datasets
from river import ensemble
from river import evaluate
from river import metrics
from river import tree

dataset = datasets.Phishing()

metric = metrics.LogLoss()

model = ensemble.AdaBoostClassifier(
    model=(
        tree.HoeffdingTreeClassifier(
            split_criterion='gini',
            delta=1e-5,
            grace_period=2000
        )
    ),
    n_models=5,
    seed=42
)

evaluate.progressive_val_score(dataset, model, metric)
```

```text
LogLoss: 0.370805
```

```python
print(model)
```

```text
AdaBoostClassifier(HoeffdingTreeClassifier)
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
> - **x**
> - **kwargs**
>
> **Returns**
>
> A dictionary that associates a probability which each label.

## References

- [Oza, N.C., 2005, October. Online bagging and boosting. In 2005 IEEE international conference on systems, man and cybernetics (Vol. 3, pp. 2340-2345). Ieee.](<https://ti.arc.nasa.gov/m/profile/oza/files/ozru01a.pdf>)
