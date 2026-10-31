# HoeffdingAdaptiveTreeRegressor

Hoeffding Adaptive Tree regressor (HATR).

This class implements a regression version of the Hoeffding Adaptive Tree Classifier. Hence, it also uses an ADWIN concept-drift detector instance at each decision node to monitor possible changes in the data distribution. If a drift is detected in a node, an alternate tree begins to be induced in the background. When enough information is gathered, HATR swaps the node where the change was detected by its alternate tree.

## Parameters

- **grace\_period** *(mutable)*

  *Type* → `int`

  *Default* → `200`

  Number of instances a leaf should observe between split attempts.
- **max\_depth**

  *Type* → `int | None`

  *Default* → `None`

  The maximum depth a tree can reach. If `None`, the tree will grow until the system recursion limit.
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

  *Default* → `adaptive`

  Prediction mechanism used at leafs. - 'mean' - Target mean - 'model' - Uses the model defined in `leaf_model` - 'adaptive' - Chooses between 'mean' and 'model' dynamically
- **leaf\_model**

  *Type* → `base.Regressor | None`

  *Default* → `None`

  The regression model used to provide responses if `leaf_prediction='model'`. If not provided an instance of `river.linear_model.LinearRegression` with the default hyperparameters is used.
- **model\_selector\_decay**

  *Type* → `float`

  *Default* → `0.95`

  The exponential decaying factor applied to the learning models' squared errors, that are monitored if `leaf_prediction='adaptive'`. Must be between `0` and `1`. The closer to `1`, the more importance is going to be given to past observations. On the other hand, if its value approaches `0`, the recent observed errors are going to have more influence on the final decision.
- **nominal\_attributes**

  *Type* → `list | None`

  *Default* → `None`

  List of Nominal attributes. If empty, then assume that all numeric attributes should be treated as continuous.
- **splitter**

  *Type* → `Splitter | None`

  *Default* → `None`

  The Splitter or Attribute Observer (AO) used to monitor the class statistics of numeric features and perform splits. Splitters are available in the `tree.splitter` module. Different splitters are available for classification and regression tasks. Classification and regression splitters can be distinguished by their property `is_target_class`. This is an advanced option. Special care must be taken when choosing different splitters. By default, `tree.splitter.TEBSTSplitter` is used if `splitter` is `None`.
- **min\_samples\_split**

  *Type* → `int`

  *Default* → `5`

  The minimum number of samples every branch resulting from a split candidate must have to be considered valid.
- **bootstrap\_sampling**

  *Type* → `bool`

  *Default* → `True`

  If True, perform bootstrap sampling in the leaf nodes.
- **drift\_window\_threshold** *(mutable)*

  *Type* → `int`

  *Default* → `300`

  Minimum number of examples an alternate tree must observe before being considered as a potential replacement to the current one.
- **drift\_detector**

  *Type* → `base.DriftDetector | None`

  *Default* → `None`

  The drift detector used to build the tree. If `None` then `drift.ADWIN` is used. Only detectors that support arbitrarily valued continuous data can be used for regression.
- **switch\_significance** *(mutable)*

  *Type* → `float`

  *Default* → `0.05`

  The significance level to assess whether alternate subtrees are significantly better than their main subtree counterparts.
- **binary\_split**

  *Type* → `bool`

  *Default* → `False`

  If True, only allow binary splits.
- **max\_size**

  *Type* → `float`

  *Default* → `500.0`

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
- **seed**

  *Type* → `int | None`

  *Default* → `None`

  Random seed for reproducibility.

## Attributes

- **height**
- **leaf\_prediction**

  Return the prediction strategy used by the tree at its leaves.
- **max\_size**

  Max allowed size tree can reach (in MiB).
- **n\_active\_leaves**
- **n\_alternate\_trees**
- **n\_branches**
- **n\_inactive\_leaves**
- **n\_leaves**
- **n\_nodes**
- **n\_pruned\_alternate\_trees**
- **n\_switch\_alternate\_trees**
- **split\_criterion**

  Return a string with the name of the split criterion being used by the tree.
- **summary**

  Collect metrics corresponding to the current status of the tree in a string buffer.

## Examples

```python
from river import datasets
from river import evaluate
from river import metrics
from river import tree
from river import preprocessing

dataset = datasets.TrumpApproval()

model = (
    preprocessing.StandardScaler() |
    tree.HoeffdingAdaptiveTreeRegressor(
        grace_period=50,
        model_selector_decay=0.3,
        seed=0
    )
)

metric = metrics.MAE()

evaluate.progressive_val_score(dataset, model, metric)
```

```text
MAE: 0.917576
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
> Train the tree model on sample x and corresponding target y.
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
> Predict the target value using one of the leaf prediction strategies.
>
> **Parameters**
>
> - **x**
>
> **Returns**
>
> Predicted target value.

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

The Hoeffding Adaptive Tree [^/dev/api/tree/HoeffdingAdaptiveTreeRegressor/#1] uses drift detectors to monitor performance of branches in the tree and to replace them with new branches when their accuracy decreases.

The bootstrap sampling strategy is an improvement over the original Hoeffding Adaptive Tree algorithm. It is enabled by default since, in general, it results in better performance.

To cope with ADWIN's requirements of bounded input data, HATR uses a novel error normalization strategy based on the empiral rule of Gaussian distributions. We assume the deviations of the predictions from the expected values follow a normal distribution. Hence, we subject these errors to a min-max normalization assuming that most of the data lies in the \\(\\left\\\[-3\\sigma, 3\\sigma\\right\\\]\\) range. These normalized errors are passed to the ADWIN instances. This is the same strategy used by Adaptive Random Forest Regressor.

[^/dev/api/tree/HoeffdingAdaptiveTreeRegressor/#1]: Bifet, Albert, and Ricard Gavaldà. "Adaptive learning from evolving data streams." In International Symposium on Intelligent Data Analysis, pp. 249-260. Springer, Berlin, Heidelberg, 2009.
