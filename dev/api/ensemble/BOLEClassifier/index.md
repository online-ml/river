# BOLEClassifier

Boosting Online Learning Ensemble (BOLE).

A modified version of Oza Online Boosting Algorithm [^/dev/api/ensemble/BOLEClassifier/#1]. For each incoming observation, each model's `learn_one` method is called `k` times where `k` is sampled from a Poisson distribution of parameter lambda. The first model to be trained will be the one with worst `correct_weight / (correct_weight + wrong_weight)`. The worst model not yet trained will receive lambda values for training from the models that incorrectly classified an instance, and the best model's not yet trained will receive lambda values for training from the models that correctly classified an instance. For more details, see [^/dev/api/ensemble/BOLEClassifier/#2].

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
- **error\_bound**

  *Default* → `0.5`

  Error bound percentage for allowing models to vote.

## Attributes

- **models**

## Examples

```python
from river import datasets
from river import ensemble
from river import evaluate
from river import drift
from river import metrics
from river import tree

dataset = datasets.Elec2().take(3000)

model = ensemble.BOLEClassifier(
    model=drift.DriftRetrainingClassifier(
        model=tree.HoeffdingTreeClassifier(),
        drift_detector=drift.binary.DDM()
    ),
    n_models=10,
    seed=42
)

metric = metrics.Accuracy()

evaluate.progressive_val_score(dataset, model, metric)
```

```text
Accuracy: 94.13%
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

[^/dev/api/ensemble/BOLEClassifier/#1]: [Oza, N.C., 2005, October. Online bagging and boosting. In 2005 IEEE international conference on systems, man and cybernetics (Vol. 3, pp. 2340-2345). Ieee.](<https://ti.arc.nasa.gov/m/profile/oza/files/ozru01a.pdf>)

[^/dev/api/ensemble/BOLEClassifier/#2]: R. S. M. d. Barros, S. Garrido T. de Carvalho Santos and P. M. Gonçalves Júnior, "A Boosting-like Online Learning Ensemble," 2016 International Joint Conference on Neural Networks (IJCNN), 2016, pp. 1871-1878, doi: 10.1109/IJCNN.2016.7727427.
