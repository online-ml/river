# iter\_evaluate

Evaluates the performance of a forecaster on a time series dataset and yields results.

This does exactly the same as [`evaluate.progressive_val_score`](<https://riverml.xyz/dev/api/evaluate/progressive-val-score>). The only difference is that this function returns an iterator, yielding results at every step. This can be useful if you want to have control over what you do with the results. For instance, you might want to plot the results.

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
