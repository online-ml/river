# ExtremelyFastDecisionTreeClassifier

Extremely Fast Decision Tree (EFDT) classifier.

Also referred to as the Hoeffding AnyTime Tree (HATT) classifier. In practice, despite the name, EFDTs are typically slower than a vanilla Hoeffding Tree to process data. The speed differences come from the mechanism of split re-evaluation present in EFDT. Nonetheless, EFDT has theoretical properties that ensure it converges faster than the vanilla Hoeffding Tree to the structure that would be created by a batch decision tree model (such as Classification and Regression Trees - CART). Keep in mind that such propositions hold when processing a stationary data stream. When dealing with non-stationary data, EFDT is somewhat robust to concept drifts as it continually revisits and updates its internal decision tree structure. Still, in such cases, the Hoeffind Adaptive Tree might be a better option, as it was specifically designed to handle non-stationarity.

## Parameters

- **grace\_period** *(mutable)*

  *Type* → `int`

  *Default* → `200`

  Number of instances a leaf should observe between split attempts.
- **max\_depth**

  *Type* → `int | None`

  *Default* → `None`

  The maximum depth a tree can reach. If `None`, the tree will grow until the system recursion limit.
- **min\_samples\_reevaluate** *(mutable)*

  *Type* → `int`

  *Default* → `20`

  Number of instances a node should observe before reevaluating the best split.
- **split\_criterion**

  *Type* → `str`

  *Default* → `info_gain`

  Split criterion to use. - 'gini' - Gini - 'info\_gain' - Information Gain - 'hellinger' - Helinger Distance
- **delta** *(mutable)*

  *Type* → `float`

  *Default* → `1e-07`

  Significance level to calculate the Hoeffding bound. The significance level is given by `1 - delta`. Values closer to zero imply longer split decision delays.
- **tau** *(mutable)*

  *Type* → `float`

  *Default* → `0.05`

  Threshold below which a split will be forced to break ties.
- **leaf\_prediction**

  *Type* → `str`

  *Default* → `nba`

  Prediction mechanism used at leafs. - 'mc' - Majority Class - 'nb' - Naive Bayes - 'nba' - Naive Bayes Adaptive
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

  The max size of the tree, in mebibytes (MiB).
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

gen = synth.Agrawal(classification_function=0, seed=42)
dataset = iter(gen.take(1000))

model = tree.ExtremelyFastDecisionTreeClassifier(
    grace_period=100,
    delta=1e-5,
    nominal_attributes=['elevel', 'car', 'zipcode'],
    min_samples_reevaluate=100
)

metric = metrics.Accuracy()

evaluate.progressive_val_score(dataset, model, metric)
```

```text
Accuracy: 87.29%
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
> Incrementally train the model
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

## Notes

The Extremely Fast Decision Tree (EFDT) [^/dev/api/tree/ExtremelyFastDecisionTreeClassifier/#1] constructs a tree incrementally. The EFDT seeks to select and deploy a split as soon as it is confident the split is useful, and then revisits that decision, replacing the split if it subsequently becomes evident that a better split is available. The EFDT learns rapidly from a stationary distribution and eventually it learns the asymptotic batch tree if the distribution from which the data are drawn is stationary.

[^/dev/api/tree/ExtremelyFastDecisionTreeClassifier/#1]: C. Manapragada, G. Webb, and M. Salehi. Extremely Fast Decision Tree. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery &amp; Data Mining (KDD '18). ACM, New York, NY, USA, 1953-1962. DOI: https://doi.org/10.1145/3219819.3220005
