# SRPRegressor

Streaming Random Patches ensemble regressor.

The Streaming Random Patches [^/dev/api/ensemble/SRPRegressor/#1] ensemble method for regression trains each base learner on a subset of features and instances from the original data, namely a random patch. This strategy to enforce diverse base models is similar to the one in the random forest, yet it is not restricted to using decision trees as base learner.

This method is an adaptation of [^/dev/api/ensemble/SRPRegressor/#2] for regression.

## Parameters

- **model**

  *Type* → `base.Regressor | None`

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

  Lambda value for bagging.
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

  *Default* → `True`

  If True, disables weighted voting.
- **drift\_detection\_criteria**

  *Type* → `str`

  *Default* → `error`

  The criteria used to track drifts.  
  \* 'error' - absolute error.  
  \* 'prediction' - predicted target values.
- **aggregation\_method**

  *Type* → `str`

  *Default* → `mean`

  The method to use to aggregate predictions in the ensemble.  
  \* 'mean'  
  \* 'median'
- **seed**

  *Default* → `None`

  Random number generator seed for reproducibility.
- **metric**

  *Type* → `RegressionMetric | None`

  *Default* → `None`

  The metric to track members performance within the ensemble.

## Attributes

- **models**

## Examples

```python
from river import ensemble
from river import evaluate
from river import metrics
from river.datasets import synth
from river import tree

dataset = synth.FriedmanDrift(
    drift_type='gsg',
    position=(350, 750),
    transition_window=200,
    seed=42
).take(1000)

base_model = tree.HoeffdingTreeRegressor(grace_period=50)
model = ensemble.SRPRegressor(
    model=base_model,
    training_method="patches",
    n_models=3,
    seed=42
)

metric = metrics.R2()

evaluate.progressive_val_score(dataset, model, metric)
```

```text
R2: 0.571117
```

## Methods

> [!NOTE]
>
> **learn\_one**

> [!NOTE]
>
> **predict\_one**
>
> Predict the output of features `x`.
>
> **Parameters**
>
> - **x**
> - **kwargs**
>
> **Returns**
>
> The prediction.

> [!NOTE]
>
> **reset**

## Notes

This implementation uses `n_models=10` as default given the impact on processing time. The optimal number of models depends on the data and resources available.

[^/dev/api/ensemble/SRPRegressor/#1]: Heitor Gomes, Jacob Montiel, Saulo Martiello Mastelini, Bernhard Pfahringer, and Albert Bifet. On Ensemble Techniques for Data Stream Regression. IJCNN'20. International Joint Conference on Neural Networks. 2020.

[^/dev/api/ensemble/SRPRegressor/#2]: Heitor Murilo Gomes, Jesse Read, Albert Bifet. Streaming Random Patches for Evolving Data Stream Classification. IEEE International Conference on Data Mining (ICDM), 2019.
