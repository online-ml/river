# SRPClassifier

Streaming Random Patches ensemble classifier.

The Streaming Random Patches (SRP) [^/dev/api/ensemble/SRPClassifier/#1] is an ensemble method that simulates bagging or random subspaces. The default algorithm uses both bagging and random subspaces, namely Random Patches. The default base estimator is a Hoeffding Tree, but other base estimators can be used (differently from random forest variations).

## Parameters

- **model**

  *Type* → `base.Estimator | None`

  *Default* → `None`

  The base estimator.
- **n\_models**

  *Type* → `int`

  *Default* → `10`

  Number of members in the ensemble.
- **subspace\_size**

  *Type* → `int | float | str`

  *Default* → `0.6`

  Number of features per subset for each classifier where `M` is the total number of features.  
  A negative value means `M - subspace_size`.  
  Only applies when using random subspaces or random patches.  
  \* If `int` indicates the number of features to use. Valid range \[2, M\].  
  \* If `float` indicates the percentage of features to use, Valid range (0., 1.\].  
  \* 'sqrt' - `sqrt(M)+1`  
  \* 'rmsqrt' - Residual from `M-(sqrt(M)+1)`
- **training\_method**

  *Type* → `str`

  *Default* → `patches`

  The training method to use.  
  \* 'subspaces' - Random subspaces.  
  \* 'resampling' - Resampling.  
  \* 'patches' - Random patches.
- **lam**

  *Type* → `int`

  *Default* → `6`

  Lambda value for resampling.
- **drift\_detector**

  *Type* → `base.DriftDetector | None`

  *Default* → `None`

  Drift detector.
- **warning\_detector**

  *Type* → `base.DriftDetector | None`

  *Default* → `None`

  Warning detector.
- **disable\_detector**

  *Type* → `str`

  *Default* → `off`

  Option to disable drift detectors:  
  \* If `'off'`, detectors are enabled.  
  \* If `'drift'`, disables concept drift detection and the background learner.  
  \* If `'warning'`, disables the background learner and ensemble members are reset if drift is detected.
- **disable\_weighted\_vote**

  *Type* → `bool`

  *Default* → `False`

  If True, disables weighted voting.
- **seed**

  *Type* → `int | None`

  *Default* → `None`

  Random number generator seed for reproducibility.
- **metric**

  *Type* → `ClassificationMetric | None`

  *Default* → `None`

  The metric to track members performance within the ensemble. This implementation assumes that larger values are better when using weighted votes.

## Attributes

- **models**

## Examples

```python
from river import ensemble
from river import evaluate
from river import metrics
from river.datasets import synth
from river import tree

dataset = synth.ConceptDriftStream(
    seed=42,
    position=500,
    width=50
).take(1000)

base_model = tree.HoeffdingTreeClassifier(
    grace_period=50, delta=0.01,
    nominal_attributes=['age', 'car', 'zipcode']
)
model = ensemble.SRPClassifier(
    model=base_model, n_models=3, seed=42,
)

metric = metrics.Accuracy()

evaluate.progressive_val_score(dataset, model, metric)
```

```text
Accuracy: 72.67%
```

## Methods

> [!NOTE]
>
> **learn\_one**

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

> [!NOTE]
>
> **reset**

## Notes

This implementation uses `n_models=10` as default given the impact on processing time. The optimal number of models depends on the data and resources available.

[^/dev/api/ensemble/SRPClassifier/#1]: Heitor Murilo Gomes, Jesse Read, Albert Bifet. Streaming Random Patches for Evolving Data Stream Classification. IEEE International Conference on Data Mining (ICDM), 2019.
