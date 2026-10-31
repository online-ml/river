# AMRules

Adaptive Model Rules.

AMRules[^/dev/api/rules/AMRules/#1] is a rule-based algorithm for incremental regression tasks. AMRules relies on the Hoeffding bound to build its rule set, similarly to Hoeffding Trees. The Variance-Ratio heuristic is used to evaluate rules' splits. Moreover, this rule-based regressor has additional capacities not usually found in decision trees.

Firstly, each created decision rule has a built-in drift detection mechanism. Every time a drift is detected, the affected decision rule is removed. In addition, AMRules' rules also have anomaly detection capabilities. After a warm-up period, each rule tests whether or not the incoming instances are anomalies. Anomalous instances are not used for training.

Every time no rule is covering an incoming example, a default rule is used to learn from it. A rule covers an instance when all of the rule's literals (tests joined by the logical operation `and`) match the input case. The default rule is also applied for predicting examples not covered by any rules from the rule set.

## Parameters

- **n\_min** *(mutable)*

  *Type* → `int`

  *Default* → `200`

  The total weight that must be observed by a rule between expansion attempts.
- **delta** *(mutable)*

  *Type* → `float`

  *Default* → `1e-07`

  The split test significance. The split confidence is given by `1 - delta`.
- **tau** *(mutable)*

  *Type* → `float`

  *Default* → `0.05`

  The tie-breaking threshold.
- **pred\_type**

  *Type* → `str`

  *Default* → `adaptive`

  The prediction strategy used by the decision rules. Can be either: - `"mean"`: outputs the target mean within the partitions defined by the decision rules. - `"model"`: always use instances of the model passed `pred_model` to make predictions. - `"adaptive"`: dynamically selects between "mean" and "model" for each incoming example. The most accurate option at the moment will be used.
- **pred\_model**

  *Type* → `base.Regressor | None`

  *Default* → `None`

  The regression model that will be replicated for every rule when `pred_type` is either `"model"` or `"adaptive"`.
- **splitter**

  *Type* → `spl.Splitter | None`

  *Default* → `None`

  The Splitter or Attribute Observer (AO) used to monitor the class statistics of numeric features and perform splits. Splitters are available in the `tree.splitter` module. Different splitters are available for classification and regression tasks. Classification and regression splitters can be distinguished by their property `is_target_class`. This is an advanced option. Special care must be taken when choosing different splitters. By default, `tree.splitter.TEBSTSplitter` is used if `splitter` is `None`.
- **drift\_detector**

  *Type* → `base.DriftDetector | None`

  *Default* → `None`

  The drift detection model that is used by each rule. Care must be taken to avoid the triggering of too many false alarms or delaying too much the concept drift detection. By default, `drift.ADWIN` is used if `drift_detector` is `None`.
- **fading\_factor** *(mutable)*

  *Type* → `float`

  *Default* → `0.99`

  The exponential decaying factor applied to the learning models' absolute errors, that are monitored if `pred_type='adaptive'`. Must be between `0` and `1`. The closer to `1`, the more importance is going to be given to past observations. On the other hand, if its value approaches `0`, the recent observed errors are going to have more influence on the final decision.
- **anomaly\_threshold** *(mutable)*

  *Type* → `float`

  *Default* → `-0.75`

  The threshold below which instances will be considered anomalies by the rules.
- **m\_min** *(mutable)*

  *Type* → `int`

  *Default* → `30`

  The minimum total weight a rule must observe before it starts to skip anomalous instances during training.
- **ordered\_rule\_set** *(mutable)*

  *Type* → `bool`

  *Default* → `True`

  If `True`, only the first rule that covers an instance will be used for training or prediction. If `False`, all the rules covering an instance will be updated during training, and the predictions for an instance will be the average prediction of all rules covering that example.
- **min\_samples\_split**

  *Type* → `int`

  *Default* → `5`

  The minimum number of samples each partition of a binary split candidate must have to be considered valid.

## Attributes

- **n\_drifts\_detected**

  The number of detected concept drifts.

## Examples

```python
from river import datasets
from river import drift
from river import evaluate
from river import metrics
from river import preprocessing
from river import rules

dataset = datasets.TrumpApproval()

model = (
    preprocessing.StandardScaler() |
    rules.AMRules(
        delta=0.01,
        n_min=50,
        drift_detector=drift.ADWIN()
    )
)

metric = metrics.MAE()

evaluate.progressive_val_score(dataset, model, metric)
```

```text
MAE: 1.119553
```

## Methods

> [!NOTE]
>
> **anomaly\_score**
>
> Aggregated anomaly score computed using all the rules that cover the input instance.
>
> Returns the mean anomaly score, the standard deviation of the score, and the proportion of rules that cover the instance (support). If the support is zero, it means that the default rule was used (not other rule covered `x`).
>
> **Parameters**
>
> - **x**
>
> **Returns**
>
> `tuple&#91;float, float, float&#93;`: mean\_anomaly\_score, std\_anomaly\_score, support

> [!NOTE]
>
> **debug\_one**
>
> Return an explanation of how `x` is predicted
>
> **Parameters**
>
> - **x**
>
> **Returns**
>
> `str`: A representation of the rules that cover the input and their prediction.

> [!NOTE]
>
> **learn\_one**
>
> Fits to a set of features `x` and a real-valued target `y`.
>
> **Parameters**
>
> - **x** — `dict`
> - **y** — `base.typing.RegTarget`
> - **w** — `int` — defaults to `1`

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

AMRules treats all the non-numerical inputs as nominal features. All instances of `numbers.Number` will be treated as continuous, even if they represent integer categories. When using nominal features, `pred_type` should be set to "mean", otherwise errors will be thrown while trying to update the underlying rules' prediction models. Prediction strategies other than "mean" can be used, as long as the prediction model passed to `pred_model` supports nominal features.

[^/dev/api/rules/AMRules/#1]: Duarte, J., Gama, J. and Bifet, A., 2016. Adaptive model rules from high-speed data streams. ACM Transactions on Knowledge Discovery from Data (TKDD), 10(3), pp.1-22.
