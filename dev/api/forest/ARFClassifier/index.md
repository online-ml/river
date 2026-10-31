# ARFClassifier

Adaptive Random Forest classifier.

The 3 most important aspects of Adaptive Random Forest [^/dev/api/forest/ARFClassifier/#1] are:

1. inducing diversity through re-sampling
2. inducing diversity through randomly selecting subsets of features for node splits
3. drift detectors per base tree, which cause selective resets in response to drifts

It also allows training background trees, which start training if a warning is detected and replace the active tree if the warning escalates to a drift.

## Parameters

- **n\_models**

  *Type* → `int`

  *Default* → `10`

  Number of trees in the ensemble.
- **max\_features** *(mutable)*

  *Type* → `bool | str | int`

  *Default* → `sqrt`

  Max number of attributes for each node split.  
  \- If `int`, then consider `max_features` at each split.  
  \- If `float`, then `max_features` is a percentage and `int(max_features * n_features)` features are considered per split.  
  \- If "sqrt", then `max_features=sqrt(n_features)`.  
  \- If "log2", then `max_features=log2(n_features)`.  
  \- If None, then `max_features=n_features`.
- **lambda\_value** *(mutable)*

  *Type* → `int`

  *Default* → `6`

  The lambda value for bagging (lambda=6 corresponds to Leveraging Bagging).
- **metric**

  *Type* → `metrics.base.MultiClassMetric | None`

  *Default* → `None`

  Metric used to track trees performance within the ensemble. Defaults to `metrics.Accuracy()`.
- **disable\_weighted\_vote**

  *Default* → `False`

  If `True`, disables the weighted vote prediction.
- **drift\_detector**

  *Type* → `base.DriftDetector | None`

  *Default* → `None`

  Drift Detection method. Set to None to disable Drift detection. Defaults to `drift.ADWIN(delta=0.001)`.
- **warning\_detector**

  *Type* → `base.DriftDetector | None`

  *Default* → `None`

  Warning Detection method. Set to None to disable warning detection. Defaults to `drift.ADWIN(delta=0.01)`.
- **grace\_period** *(mutable)*

  *Type* → `int`

  *Default* → `50`

  \[*Tree parameter*\] Number of instances a leaf should observe between split attempts.
- **max\_depth**

  *Type* → `int | None`

  *Default* → `None`

  \[*Tree parameter*\] The maximum depth a tree can reach. If `None`, the tree will grow until the system recursion limit.
- **split\_criterion**

  *Type* → `str`

  *Default* → `info_gain`

  \[*Tree parameter*\] Split criterion to use.  
  \- 'gini' - Gini  
  \- 'info\_gain' - Information Gain  
  \- 'hellinger' - Hellinger Distance
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

  *Default* → `nba`

  \[*Tree parameter*\] Prediction mechanism used at leafs.  
  \- 'mc' - Majority Class  
  \- 'nb' - Naive Bayes  
  \- 'nba' - Naive Bayes Adaptive
- **nb\_threshold**

  *Type* → `int`

  *Default* → `0`

  \[*Tree parameter*\] Number of instances a leaf should observe before allowing Naive Bayes.
- **nominal\_attributes**

  *Type* → `list | None`

  *Default* → `None`

  \[*Tree parameter*\] List of Nominal attributes. If empty, then assume that all attributes are numerical.
- **splitter**

  *Type* → `Splitter | None`

  *Default* → `None`

  \[*Tree parameter*\] The Splitter or Attribute Observer (AO) used to monitor the class statistics of numeric features and perform splits. Splitters are available in the `tree.splitter` module. Different splitters are available for classification and regression tasks. Classification and regression splitters can be distinguished by their property `is_target_class`. This is an advanced option. Special care must be taken when choosing different splitters. By default, `tree.splitter.GaussianSplitter` is used if `splitter` is `None`.
- **binary\_split**

  *Type* → `bool`

  *Default* → `False`

  \[*Tree parameter*\] If True, only allow binary splits.
- **min\_branch\_fraction**

  *Type* → `float`

  *Default* → `0.01`

  \[*Tree parameter*\] The minimum percentage of observed data required for branches resulting from split candidates. To validate a split candidate, at least two resulting branches must have a percentage of samples greater than `min_branch_fraction`. This criterion prevents unnecessary splits when the majority of instances are concentrated in a single branch.
- **max\_share\_to\_split**

  *Type* → `float`

  *Default* → `0.99`

  \[*Tree parameter*\] Only perform a split in a leaf if the proportion of elements in the majority class is smaller than this parameter value. This parameter avoids performing splits when most of the data belongs to a single class.
- **max\_size**

  *Type* → `float`

  *Default* → `100.0`

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

## Examples

```python
from river import evaluate
from river import forest
from river import metrics
from river.datasets import synth

dataset = synth.ConceptDriftStream(
    seed=42,
    position=500,
    width=40
).take(1000)

model = forest.ARFClassifier(seed=8, leaf_prediction="mc")

metric = metrics.Accuracy()

evaluate.progressive_val_score(dataset, model, metric)
```

```text
Accuracy: 67.97%
```

The total number of warnings and drifts detected, respectively

```python
model.n_warnings_detected(), model.n_drifts_detected()
```

```text
(2, 1)
```

The number of warnings detected by tree number 2

```python
model.n_warnings_detected(2)
```

```text
1
```

And the corresponding number of actual concept drift detected

```python
model.n_drifts_detected(2)
```

```text
1
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
> - **x** — `dict`
> - **kwargs** — `typing.Any`
>
> **Returns**
>
> `dict&#91;base.typing.ClfTarget, float&#93;`: A dictionary that associates a probability which each label.

[^/dev/api/forest/ARFClassifier/#1]: Heitor Murilo Gomes, Albert Bifet, Jesse Read, Jean Paul Barddal, Fabricio Enembreck, Bernhard Pfharinger, Geoff Holmes, Talel Abdessalem. Adaptive random forests for evolving data stream classification. In Machine Learning, DOI: 10.1007/s10994-017-5642-8, Springer, 2017.
