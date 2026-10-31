# ISOUPTreeRegressor

Incremental Structured Output Prediction Tree (iSOUP-Tree) for multi-target regression.

This is an implementation of the iSOUP-Tree proposed by A. Osojnik, P. Panov, and S. Džeroski [^/dev/api/tree/ISOUPTreeRegressor/#1].

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

  Allowed error in split decision, a value closer to 0 takes longer to decide.
- **tau** *(mutable)*

  *Type* → `float`

  *Default* → `0.05`

  Threshold below which a split will be forced to break ties.
- **leaf\_prediction**

  *Type* → `str`

  *Default* → `adaptive`

  Prediction mechanism used at leafs. - 'mean' - Target mean - 'model' - Uses the model defined in `leaf_model` - 'adaptive' - Chooses between 'mean' and 'model' dynamically
- **leaf\_model**

  *Type* → `base.Regressor | dict | None`

  *Default* → `None`

  The regression model(s) used to provide responses if `leaf_prediction='model'`. It can be either a regressor (in which case it is going to be replicated to all the targets) or a dictionary whose keys are target identifiers, and the values are instances of `river.base.Regressor.` If not provided, instances of `river.linear_model.LinearRegression` with the default hyperparameters are used for all the targets. If a dictionary is passed and not all target models are specified, copies from the first model match in the dictionary will be used to the remaining targets.
- **model\_selector\_decay**

  *Type* → `float`

  *Default* → `0.95`

  The exponential decaying factor applied to the learning models' squared errors, that are monitored if `leaf_prediction='adaptive'`. Must be between `0` and `1`. The closer to `1`, the more importance is going to be given to past observations. On the other hand, if its value approaches `0`, the recent observed errors are going to have more influence on the final decision.
- **nominal\_attributes**

  *Type* → `list | None`

  *Default* → `None`

  List of Nominal attributes identifiers. If empty, then assume that all numeric attributes should be treated as continuous.
- **splitter**

  *Type* → `Splitter | None`

  *Default* → `None`

  The Splitter or Attribute Observer (AO) used to monitor the class statistics of numeric features and perform splits. Splitters are available in the `tree.splitter` module. Different splitters are available for classification and regression tasks. Classification and regression splitters can be distinguished by their property `is_target_class`. This is an advanced option. Special care must be taken when choosing different splitters. By default, `tree.splitter.TEBSTSplitter` is used if `splitter` is `None`.
- **min\_samples\_split**

  *Type* → `int`

  *Default* → `5`

  The minimum number of samples every branch resulting from a split candidate must have to be considered valid.
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
import numbers
from river import compose
from river import datasets
from river import evaluate
from river import linear_model
from river import metrics
from river import preprocessing
from river import tree

dataset = datasets.SolarFlare()

num = compose.SelectType(numbers.Number) | preprocessing.MinMaxScaler()
cat = compose.SelectType(str) | preprocessing.OneHotEncoder()

model = tree.ISOUPTreeRegressor(
    grace_period=100,
    leaf_prediction='model',
    leaf_model={
        'c-class-flares': linear_model.LinearRegression(l2=0.02),
        'm-class-flares': linear_model.PARegressor(),
        'x-class-flares': linear_model.LinearRegression(l2=0.1)
    }
)

pipeline = (num + cat) | model
metric = metrics.multioutput.MicroAverage(metrics.MAE())

evaluate.progressive_val_score(dataset, pipeline, metric)
```

```text
MicroAverage(MAE): 0.426177
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
> Incrementally train the model with one sample.
>
> Training tasks: \* If the tree is empty, create a leaf node as the root. \* If the tree is already initialized, find the corresponding leaf for the instance and update the leaf node statistics. \* If growth is allowed and the number of instances that the leaf has observed between split attempts exceed the grace period then attempt to split.
>
> **Parameters**
>
> - **x**
> - **y**
> - **w** — `float` — defaults to `1.0`
> - **kwargs**

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

[^/dev/api/tree/ISOUPTreeRegressor/#1]: Aljaž Osojnik, Panče Panov, and Sašo Džeroski. "Tree-based methods for online multi-target regression." Journal of Intelligent Information Systems 50.2 (2018): 315-339.
