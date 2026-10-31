# LASTClassifier

Local Adaptive Streaming Tree Classifier.

Local Adaptive Streaming Tree [^/dev/api/tree/LASTClassifier/#1] (LAST) is an incremental decision tree with adaptive splitting mechanisms. LAST maintains a change detector at each leaf and splits this node if a change is detected in the error or the leaf\`s data distribution.

LAST is still not suitable for use as a base classifier in ensembles due to the change detectors. The authors in [^/dev/api/tree/LASTClassifier/#1] are working on a version of LAST that overcomes this limitation.

## Parameters

- **max\_depth**

  *Type* → `int | None`

  *Default* → `None`

  The maximum depth a tree can reach. If `None`, the tree will grow until the system recursion limit.
- **split\_criterion**

  *Type* → `str`

  *Default* → `info_gain`

  Split criterion to use. - 'gini' - Gini - 'info\_gain' - Information Gain - 'hellinger' - Helinger Distance
- **leaf\_prediction**

  *Type* → `str`

  *Default* → `nba`

  Prediction mechanism used at leafs. - 'mc' - Majority Class - 'nb' - Naive Bayes - 'nba' - Naive Bayes Adaptive
- **change\_detector**

  *Type* → `base.DriftDetector | None`

  *Default* → `None`

  Change detector that will be created at each leaf of the tree.
- **track\_error**

  *Type* → `bool`

  *Default* → `True`

  If True, the change detector will have binary inputs for error predictions, otherwise the input will be the split criteria.
- **nb\_threshold**

  *Type* → `int`

  *Default* → `0`

  Number of instances a leaf should observe before allowing Naive Bayes.
- **nominal\_attributes**

  *Type* → `list | None`

  *Default* → `None`

  List of Nominal attributes identifiers. If empty, then assume that all numeric attributes should be treated as continuous.
- **splitter**

  *Type* → `Splitter | None`

  *Default* → `None`

  The Splitter or Attribute Observer (AO) used to monitor the class statistics of numeric features and perform splits. Splitters are available in the `tree.splitter` module. Different splitters are available for classification and regression tasks. Classification and regression splitters can be distinguished by their property `is_target_class`. This is an advanced option. Special care must be taken when choosing different splitters. By default, `tree.splitter.GaussianSplitter` is used if `splitter` is `None`.
- **binary\_split**

  *Type* → `bool`

  *Default* → `False`

  If True, only allow binary splits.
- **min\_branch\_fraction**

  *Type* → `float`

  *Default* → `0.01`

  The minimum percentage of observed data required for branches resulting from split candidates. To validate a split candidate, at least two resulting branches must have a percentage of samples greater than `min_branch_fraction`. This criterion prevents unnecessary splits when the majority of instances are concentrated in a single branch.
- **max\_share\_to\_split**

  *Type* → `float`

  *Default* → `0.99`

  Only perform a split in a leaf if the proportion of elements in the majority class is smaller than this parameter value. This parameter avoids performing splits when most of the data belongs to a single class.
- **max\_size**

  *Type* → `float`

  *Default* → `100.0`

  The max size of the tree, in Megabytes (MB).
- **memory\_estimate\_period**

  *Type* → `int`

  *Default* → `1000000`

  Interval (number of processed instances) between memory consumption checks.
- **stop\_mem\_management**

  *Type* → `bool`

  *Default* → `False`

  If True, stop growing as soon as memory limit is hit.
- **remove\_poor\_attrs**

  *Type* → `bool`

  *Default* → `False`

  If True, disable poor attributes to reduce memory usage.
- **merit\_preprune**

  *Type* → `bool`

  *Default* → `True`

  If True, enable merit-based tree pre-pruning.

## Attributes

- **height**
- **leaf\_prediction**

  Return the prediction strategy used by the tree at its leaves.
- **max\_size**

  Max allowed size tree can reach (in MiB).
- **n\_active\_leaves**
- **n\_branches**
- **n\_inactive\_leaves**
- **n\_leaves**
- **n\_nodes**
- **split\_criterion**

  Return a string with the name of the split criterion being used by the tree.
- **summary**

  Collect metrics corresponding to the current status of the tree in a string buffer.

## Examples

```python
from river.datasets import synth
from river import evaluate
from river import metrics
from river import tree

gen = synth.ConceptDriftStream(stream=synth.SEA(seed=42, variant=0),
                       drift_stream=synth.SEA(seed=42, variant=1),
                       seed=1, position=1500, width=50)
dataset = iter(gen.take(3000))

model = tree.LASTClassifier()

metric = metrics.Accuracy()

evaluate.progressive_val_score(dataset, model, metric)
```

```text
Accuracy: 91.10%
```

## Methods

> [!NOTE]
>
> **debug\_one**
>
> Print an explanation of how `x` is predicted.
>
> **Parameters**
>
> - **x** — `dict`
>
> **Returns**
>
> `str | None`: A representation of the path followed by the tree to predict `x`; `None` if

> [!NOTE]
>
> **draw**
>
> Draw the tree using the `graphviz` library.
>
> Since the tree is drawn without passing incoming samples, classification trees will show the majority class in their leaves, whereas regression trees will use the target mean.
>
> **Parameters**
>
> - **max\_depth** — `int | None` — defaults to `None`  
>   The maximum depth a tree can reach. If `None`, the tree will grow until the system recursion limit.

> [!NOTE]
>
> **learn\_one**
>
> Train the model on instance x and corresponding target y.
>
> **Parameters**
>
> - **x**
> - **y**
> - **w** — defaults to `1.0`

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
>
> **Returns**
>
> A dictionary that associates a probability which each label.

> [!NOTE]
>
> **to\_dataframe**
>
> Return a representation of the current tree structure organized in a `pandas.DataFrame` object.
>
> In case the tree is empty or it only contains a single node (a leaf), `None` is returned.
>
> **Returns**
>
> df

[^/dev/api/tree/LASTClassifier/#1]: Daniel Nowak Assis, Jean Paul Barddal, and Fabrício Enembreck. Just Change on Change: Adaptive Splitting Time for Decision Trees in Data Stream Classification . In Proceedings of ACM SAC Conference (SAC’24).
