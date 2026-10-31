# OXTRegressor

Online Extra Trees regressor.

The online Extra Trees[^/dev/api/forest/OXTRegressor/#1] ensemble takes some steps further into randomization when compared to Adaptive Random Forests (ARF). A subspace of the feature space is considered at each split attempt, as ARF does, and online bagging or subbagging can also be (optionally) used. Nonetheless, Extra Trees randomizes the split candidates evaluated by each leaf node (just a single split is tested by numerical feature, which brings significant speedups to the ensemble), and might also randomize the maximum depth of the forest members, as well as the size of the feature subspace processed by each of its trees' leaves.

On the other hand, OXT suffers from a cold-start problem. As the splits are random, the predictive performance in small samples is usually worse than using a deterministic split approach, such as the one used by ARF.

## Parameters

- **n\_models**

  *Type* → `int`

  *Default* → `10`

  The number of trees in the ensemble.
- **max\_features**

  *Type* → `bool | str | int`

  *Default* → `sqrt`

  Max number of attributes for each node split. - If int, then consider `max_features` at each split. - If float, then `max_features` is a percentage and `int(max_features * n_features)` features are considered per split. - If "sqrt", then `max_features=sqrt(n_features)`. - If "log2", then `max_features=log2(n_features)`. - If "random", then `max_features` will assume a different random number in the interval `\[2, n_features\]` for each tree leaf. - If None, then `max_features=n_features`.
- **resampling\_strategy**

  *Type* → `str | None`

  *Default* → `subbagging`

  The chosen instance resampling strategy: - If `None`, no resampling will be done and the trees will process all instances. - If `'baggging'`, online bagging will be performed (sampling with replacement). - If `'subbagging'`, online subbagging will be performed (sampling without replacement).
- **resampling\_rate**

  *Type* → `int | float`

  *Default* → `0.5`

  Only valid if `resampling_strategy` is not None. Controls the parameters of the resampling strategy.. - If `resampling_strategy='bagging'`, must be an integer greater than or equal to 1 that parameterizes the poisson distribution used to simulate bagging in online learning settings. It acts as the lambda parameter of Oza Bagging and Leveraging Bagging. - If `resampling_strategy='subbagging'`, must be a float in the interval \\((0, 1\]\\) that controls the chance of each instance being used by a tree for learning.
- **detection\_mode**

  *Type* → `str`

  *Default* → `all`

  The concept drift detection mode in which the forest operates. Valid values are: - "all": creates both warning and concept drift detectors. If a warning is detected, an alternate tree starts being trained in the background. If the warning trigger escalates to a concept drift, the affected tree is replaced by the alternate tree. - "drop": only the concept drift detectors are created. If a drift is detected, the affected tree is dropped and replaced by a new tree. - "off": disables the concept drift adaptation capabilities. The forest will act as if the processed stream is stationary.
- **warning\_detector**

  *Type* → `base.DriftDetector | None`

  *Default* → `None`

  The detector that will be used to trigger concept drift warnings. Defaults to `drift.ADWIN(0.01)`.
- **drift\_detector**

  *Type* → `base.DriftDetector | None`

  *Default* → `None`

  The detector used to detect concept drifts. Defaults to `drift.ADWIN(0.001)`.
- **max\_depth**

  *Type* → `int | None`

  *Default* → `None`

  The maximum depth the ensemble members might reach. If `None`, the trees will grow indefinitely.
- **randomize\_tree\_depth**

  *Type* → `bool`

  *Default* → `False`

  Whether or not randomize the maximum depth of each tree in the ensemble. If `max_depth` is provided, it is going to act as an upper bound to generate the maximum depth for each tree.
- **track\_metric**

  *Type* → `metrics.base.RegressionMetric | None`

  *Default* → `None`

  The performance metric used to weight predictions. Defaults to `metrics.MAE()`.
- **disable\_weighted\_vote**

  *Type* → `bool`

  *Default* → `True`

  Defines whether or not to use predictions weighted by each trees' prediction performance.
- **split\_buffer\_size**

  *Type* → `int`

  *Default* → `5`

  Defines the size of the buffer used by the tree splitters when determining the feature range and a random split point in this interval.
- **seed**

  *Type* → `int | None`

  *Default* → `None`

  Random seed to support reproducibility.
- **grace\_period**

  *Type* → `int`

  *Default* → `50`

  \[*Tree parameter*\] Number of instances a leaf should observe between split attempts.
- **delta**

  *Type* → `float`

  *Default* → `0.01`

  \[*Tree parameter*\] Allowed error in split decision, a value closer to 0 takes longer to decide.
- **tau**

  *Type* → `float`

  *Default* → `0.05`

  \[*Tree parameter*\] Threshold below which a split will be forced to break ties.
- **leaf\_prediction**

  *Type* → `str`

  *Default* → `adaptive`

  \[*Tree parameter*\] Prediction mechanism used at leaves. - 'mean' - Target mean - 'model' - Uses the model defined in `leaf_model` - 'adaptive' - Chooses between 'mean' and 'model' dynamically
- **leaf\_model**

  *Type* → `base.Regressor | None`

  *Default* → `None`

  \[*Tree parameter*\] The regression model used to provide responses if `leaf_prediction='model'`. If not provided, an instance of `river.linear_model.LinearRegression` with the default hyperparameters is used.
- **model\_selector\_decay**

  *Type* → `float`

  *Default* → `0.95`

  \[*Tree parameter*\] The exponential decaying factor applied to the learning models' squared errors, that are monitored if `leaf_prediction='adaptive'`. Must be between `0` and `1`. The closer to `1`, the more importance is going to be given to past observations. On the other hand, if its value approaches `0`, the recent observed errors are going to have more influence on the final decision.
- **nominal\_attributes**

  *Type* → `list | None`

  *Default* → `None`

  \[*Tree parameter*\] List of Nominal attributes. If empty, then assume that all attributes are numerical.
- **min\_samples\_split**

  *Type* → `int`

  *Default* → `5`

  \[*Tree parameter*\] The minimum number of samples every branch resulting from a split candidate must have to be considered valid.
- **binary\_split**

  *Type* → `bool`

  *Default* → `False`

  \[*Tree parameter*\] If True, only allow binary splits.
- **max\_size**

  *Type* → `int`

  *Default* → `500`

  \[*Tree parameter*\] Maximum memory (MiB) consumed by the tree.
- **memory\_estimate\_period**

  *Type* → `int`

  *Default* → `2000000`

  \[*Tree parameter*\] Number of instances between memory consumption checks.
- **stop\_mem\_management**

  *Type* → `bool`

  *Default* → `False`

  \[*Tree parameter*\] If True, stop growing as soon as memory limit is hit.
- **remove\_poor\_attrs**

  *Type* → `bool`

  *Default* → `False`

  \[*Tree parameter*\] If True, disable poor attributes to reduce memory usage.
- **merit\_preprune**

  *Type* → `bool`

  *Default* → `True`

  \[*Tree parameter*\] If True, enable merit-based tree pre-pruning.

## Attributes

- **instances\_per\_tree**

  The number of instances processed by each one of the current forest members. Each time a concept drift is detected, the count corresponding to the affected tree is reset.
- **models**
- **n\_drifts**

  The number of concept drifts detected per ensemble member.
- **n\_tree\_swaps**

  The number of performed alternate tree swaps. Not applicable if the warning detectors are disabled.
- **n\_warnings**

  The number of warnings detected per ensemble member.
- **total\_instances**

  The total number of instances processed by the ensemble.

## Examples

```python
from river import datasets
from river import evaluate
from river import metrics
from river import forest

dataset = datasets.synth.Friedman(seed=42).take(5000)

model = forest.OXTRegressor(n_models=3, seed=42)

metric = metrics.RMSE()

evaluate.progressive_val_score(dataset, model, metric)
```

```text
RMSE: 3.16212
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
> - **x** — `dict`
>
> **Returns**
>
> `base.typing.RegTarget`: The prediction.

## Notes

As the Online Extra Trees change the way in which Hoeffding Trees perform split attempts and monitor numerical input features, some of the parameters of the vanilla Hoeffding Tree algorithms are not available.

[^/dev/api/forest/OXTRegressor/#1]: Mastelini, S. M., Nakano, F. K., Vens, C., &amp; de Leon Ferreira, A. C. P. (2022). Online Extra Trees Regressor. IEEE Transactions on Neural Networks and Learning Systems.
