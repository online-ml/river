# SGTRegressor

Stochastic Gradient Tree for regression.

Incremental decision tree regressor that minimizes the mean square error to guide its growth.

Stochastic Gradient Trees (SGT) directly minimize a loss function to guide tree growth and update their predictions. Thus, they differ from other incrementally tree learners that do not directly optimize the loss, but a data impurity-related heuristic.

## Parameters

- **delta**

  *Type* → `float`

  *Default* → `1e-07`

  Define the significance level of the F-tests performed to decide upon creating splits or updating predictions.
- **grace\_period**

  *Type* → `int`

  *Default* → `200`

  Interval between split attempts or prediction updates.
- **init\_pred**

  *Type* → `float`

  *Default* → `0.0`

  Initial value predicted by the tree.
- **max\_depth**

  *Type* → `int | None`

  *Default* → `None`

  The maximum depth the tree might reach. If set to `None`, the trees will grow indefinitely.
- **lambda\_value**

  *Type* → `float`

  *Default* → `0.1`

  Positive float value used to impose a penalty over the tree's predictions and force them to become smaller. The greater the lambda value, the more constrained are the predictions.
- **gamma**

  *Type* → `float`

  *Default* → `1.0`

  Positive float value used to impose a penalty over the tree's splits and force them to be avoided when possible. The greater the gamma value, the smaller the chance of a split occurring.
- **nominal\_attributes**

  *Type* → `list | None`

  *Default* → `None`

  List with identifiers of the nominal attributes. If None, all features containing numbers are assumed to be numeric.
- **feature\_quantizer**

  *Type* → `tree.splitter.Quantizer | None`

  *Default* → `None`

  The algorithm used to quantize numeric features. Either a static quantizer (as in the original implementation) or a dynamic quantizer can be used. The correct choice and setup of the feature quantizer is a crucial step to determine the performance of SGTs. Feature quantizers are akin to the attribute observers used in Hoeffding Trees. By default, an instance of `tree.splitter.StaticQuantizer` (with default parameters) is used if this parameter is not set.

## Attributes

- **height**
- **n\_branches**
- **n\_leaves**
- **n\_node\_updates**
- **n\_nodes**
- **n\_observations**
- **n\_splits**

## Examples

```python
from river import datasets
from river import evaluate
from river import metrics
from river import tree

dataset = datasets.TrumpApproval()
model = tree.SGTRegressor(
    delta=0.01,
    lambda_value=0.01,
    grace_period=20,
    feature_quantizer=tree.splitter.DynamicQuantizer(std_prop=0.1)
)
metric = metrics.MAE()

evaluate.progressive_val_score(dataset, model, metric)
```

```text
MAE: 1.721818
```

## Methods

> [!NOTE]
>
> **learn\_one**
>
> Fits to a set of features `x` and a real-valued target `y`.
>
> **Parameters**
>
> - **x** — `dict[base.typing.FeatureName, Any]`
> - **y** — `base.typing.RegTarget`
> - **w** — defaults to `1.0`

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

This implementation enhances the original proposal [^/dev/api/tree/SGTRegressor/#1] by using an incremental strategy to discretize numerical features dynamically, rather than relying on a calibration set and parameterized number of bins. The strategy used is an adaptation of the Quantization Observer (QO) [^/dev/api/tree/SGTRegressor/#2]. Different bin size setting policies are available for selection. They directly related to number of split candidates the tree is going to explore, and thus, how accurate its split decisions are going to be. Besides, the number of stored bins per feature is directly related to the tree's memory usage and runtime.

[^/dev/api/tree/SGTRegressor/#1]: Gouk, H., Pfahringer, B., &amp; Frank, E. (2019, October). Stochastic Gradient Trees. In Asian Conference on Machine Learning (pp. 1094-1109).

[^/dev/api/tree/SGTRegressor/#2]: Mastelini, S.M. and de Leon Ferreira, A.C.P., 2021. Using dynamical quantization to perform split attempts in online tree regressors. Pattern Recognition Letters.
