# evaluate

Evaluates the performance of a forecaster on a time series dataset.

To understand why this method is useful, it's important to understand the difference between nowcasting and forecasting. Nowcasting is about predicting a value at the next time step. This can be seen as a special case of regression, where the value to predict is the value at the next time step. In this case, the [`evaluate.progressive_val_score`](<https://riverml.xyz/dev/api/evaluate/progressive-val-score>) function may be used to evaluate a model via progressive validation.

Forecasting models can also be evaluated via progressive validation. This is the purpose of this function. At each time step `t`, the forecaster is asked to predict the values at `t + 1`, `t + 2`, ..., `t + horizon`. The performance at each time step is measured and returned.

## Parameters

- **dataset**

  *Type* → `base.typing.Dataset`

  A sequential time series.
- **model**

  A forecaster.
- **metric**

  *Type* → `metrics.base.RegressionMetric`

  A regression metric.
- **horizon**

  *Type* → `int`
- **agg\_func**

  *Type* → `typing.Callable[[list[float]], float] | None`

  *Default* → `None`
