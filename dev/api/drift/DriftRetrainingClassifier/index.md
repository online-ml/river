# DriftRetrainingClassifier

Drift retraining classifier.

This classifier is a wrapper for any classifier. It monitors the incoming data for concept drifts and warnings in the model's accuracy. In case a warning is detected, a background model starts to train. If a drift is detected, the model will be replaced by the background model, and the background model will be reset.

## Parameters

- **model**

  *Type* → `base.Classifier`

  The classifier and background classifier class.
- **drift\_detector**

  *Type* → `base.DriftAndWarningDetector | base.BinaryDriftAndWarningDetector | None`

  *Default* → `None`

  Algorithm to track warnings and concept drifts. Attention! If the parameter train\_in\_background is True, the drift\_detector must have a warning tracker.
- **train\_in\_background**

  *Type* → `bool`

  *Default* → `True`

  Parameter to determine if a background model will be used.

## Examples

```python
from river import datasets
from river import evaluate
from river import drift
from river import metrics
from river import tree

dataset = datasets.Elec2().take(3000)

model = drift.DriftRetrainingClassifier(
    model=tree.HoeffdingTreeClassifier(),
    drift_detector=drift.binary.DDM()
)

metric = metrics.Accuracy()

evaluate.progressive_val_score(dataset, model, metric)
```

```text
Accuracy: 87.06%
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
> - **x** — `dict[base.typing.FeatureName, Any]`
> - **y** — `base.typing.ClfTarget`
> - **kwargs** — `Any`

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
> - **x** — `dict[base.typing.FeatureName, Any]`
> - **kwargs** — `Any`
>
> **Returns**
>
> `dict&#91;base.typing.ClfTarget, float&#93;`: A dictionary that associates a probability which each label.
