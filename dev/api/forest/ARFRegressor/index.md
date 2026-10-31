# ARFRegressor

Adaptive Random Forest regressor.

The 3 most important aspects of Adaptive Random Forest [^/dev/api/forest/ARFRegressor/#1] are:

1. inducing diversity through re-sampling
2. inducing diversity through randomly selecting subsets of features for node splits
3. drift detectors per base tree, which cause selective resets in response to drifts

Notice that this implementation is slightly different from the original algorithm proposed in [^/dev/api/forest/ARFRegressor/#2]. The `HoeffdingTreeRegressor` is used as base learner, instead of `FIMT-DD`. It also adds a new strategy to monitor the predictions and check for concept drifts. The deviations of the predictions to the target are monitored and normalized in the \[0, 1\] range to fulfill ADWIN's requirements. We assume that the data subjected to the normalization follows a normal distribution, and thus, lies within the interval of the mean \\(\\pm3\\sigma\\).

## Parameters

- **n\_models**

  *Type* → `int`

  *Default* → `10`

  Number of trees in the ensemble.
- **max\_features** *(mutable)*

  *Default* → `sqrt`

  Max number of attributes for each node split.  
  \- If `int`, then consider `max_features` at each split.  
  \- If `float`, then `max_features` is a percentage and `int(max_features * n_features)` features are considered per split.  
  \- If "sqrt", then `max_features=sqrt(n_features)`.  
  \- If "log2", then `max_features=log2(n_features)`.  
  \- If None, then `max_features=n_features`.
- **aggregation\_method** *(mutable)*

  *Type* → `str`

  *Default* → `median`

  The method to use to aggregate predictions in the ensemble.  
  \- 'mean'  
  \- 'median' - If selected will disable the weighted vote.
- **lambda\_value** *(mutable)*

  *Type* → `int`

  *Default* → `6`

  The lambda value for bagging (lambda=6 corresponds to Leveraging Bagging).
- **metric**

  *Type* → `metrics.base.RegressionMetric | None`

  *Default* → `None`

  Metric used to track trees performance within the ensemble. Depending, on the configuration, this metric is also used to weight predictions from the members of the ensemble. Defaults to `metrics.MSE()`.
- **disable\_weighted\_vote**

  *Default* → `True`

  If `True`, disables the weighted vote prediction, i.e. does not assign weights to individual tree's predictions and uses the arithmetic mean instead. Otherwise will use the `metric` value to weight predictions.
- **drift\_detector**

  *Type* → `base.DriftDetector | None`

  *Default* → `None`

  Drift Detection method. Set to None to disable Drift detection. Defaults to `drift.ADWIN(0.001)`.
- **warning\_detector**

  *Type* → `base.DriftDetector | None`

  *Default* → `None`

  Warning Detection method. Set to None to disable warning detection. Defaults to `drift.ADWIN(0.01)`.
- **grace\_period** *(mutable)*

  *Type* → `int`

  *Default* → `50`

  \[*Tree parameter*\] Number of instances a leaf should observe between split attempts.
- **max\_depth**

  *Type* → `int | None`

  *Default* → `None`

  \[*Tree parameter*\] The maximum depth a tree can reach. If `None`, the tree will grow until the system recursion limit.
- **delta** *(mutable)*

  *Type* → `float`

  *Default* → `0.01`

  \[*Tree parameter*\] Allowed error in split decision, a value closer to 0 takes longer to decide.
- **tau** *(mutable)*

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
- **model\_selector\_decay** *(mutable)*

  *Type* → `float`

  *Default* → `0.95`

  \[*Tree parameter*\] The exponential decaying factor applied to the learning models' squared errors, that are monitored if `leaf_prediction='adaptive'`. Must be between `0` and `1`. The closer to `1`, the more importance is going to be given to past observations. On the other hand, if its value approaches `0`, the recent observed errors are going to have more influence on the final decision.
- **nominal\_attributes**

  *Type* → `list | None`

  *Default* → `None`

  \[*Tree parameter*\] List of Nominal attributes. If empty, then assume that all attributes are numerical.
- **splitter**

  *Type* → `Splitter | None`

  *Default* → `None`

  \[*Tree parameter*\] The Splitter or Attribute Observer (AO) used to monitor the class statistics of numeric features and perform splits. Splitters are available in the `tree.splitter` module. Different splitters are available for classification and regression tasks. Classification and regression splitters can be distinguished by their property `is_target_class`. This is an advanced option. Special care must be taken when choosing different splitters.By default, `tree.splitter.EBSTSplitter` is used if `splitter` is `None`.
- **min\_samples\_split**

  *Type* → `int`

  *Default* → `5`

  \[*Tree parameter*\] The minimum number of samples every branch resulting from a split candidate must have to be considered valid.
- **binary\_split**

  *Type* → `bool`

  *Default* → `False`

  \[*Tree parameter*\] If True, only allow binary splits.
- **max\_size**

  *Type* → `float`

  *Default* → `500.0`

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
- **seed**

  *Type* → `int | None`

  *Default* → `None`

  Random seed for reproducibility.

## Attributes

- **models**
- **valid\_aggregation\_method**

  Valid aggregation\_method values.

## Examples

```python
from river import datasets
from river import evaluate
from river import forest
from river import metrics
from river import preprocessing

dataset = datasets.TrumpApproval()

model = (
    preprocessing.StandardScaler() |
    forest.ARFRegressor(seed=42)
)

metric = metrics.MAE()

evaluate.progressive_val_score(dataset, model, metric)
```

```text
MAE: 0.772113
```

## Methods

> [!NOTE]
>
> **learn\_one**

> [!NOTE]
>
> **n\_drifts\_detected**
>
> Get the total number of concept drifts detected, or such number on an individual tree basis (optionally).
>
> **Parameters**
>
> - **tree\_id** — `int | None` — defaults to `None`
>
> **Returns**
>
> `int`: The number of concept drifts detected.

> [!NOTE]
>
> **n\_warnings\_detected**
>
> Get the total number of concept drift warnings detected, or the number on an individual tree basis (optionally).
>
> **Parameters**
>
> - **tree\_id** — `int | None` — defaults to `None`
>
> **Returns**
>
> `int`: The number of concept drift warnings detected.

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

[^/dev/api/forest/ARFRegressor/#1]: Gomes, H.M., Bifet, A., Read, J., Barddal, J.P., Enembreck, F., Pfharinger, B., Holmes, G. and Abdessalem, T., 2017. Adaptive random forests for evolving data stream classification. Machine Learning, 106(9-10), pp.1469-1495.

[^/dev/api/forest/ARFRegressor/#2]: Gomes, H.M., Barddal, J.P., Boiko, L.E., Bifet, A., 2018. Adaptive random forests for data stream regression. ESANN 2018.
