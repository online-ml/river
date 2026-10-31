# RegressionJackknife

Jackknife method for regression.

This is a conformal prediction method for regression. It is based on the jackknife method. The idea is to compute the quantiles of the residuals of the regressor. The prediction interval is then computed as the prediction of the regressor plus the quantiles of the residuals.

This works naturally online, as the quantiles of the residuals are updated at each iteration. Each residual is produced before the regressor is updated, which ensures the predicted intervals are not optimistic.

Note that the produced intervals are marginal and not conditional. This means that the intervals are not adjusted for the features `x`. This is a limitation of the jackknife method. However, the jackknife method is very simple and efficient. It is also very robust to outliers.

## Parameters

- **regressor**

  *Type* → `T`

  The regressor to be wrapped.
- **confidence\_level**

  *Type* → `float`

  *Default* → `0.95`

  The confidence level of the prediction intervals.
- **window\_size**

  *Type* → `int | None`

  *Default* → `None`

  The size of the window used to compute the quantiles of the residuals. If `None`, the quantiles are computed over the whole history. It is advised to set this if you expect the model's performance to change over time.

## Examples

```python
from river import conf
from river import datasets
from river import linear_model
from river import metrics
from river import preprocessing
from river import stats

dataset = datasets.TrumpApproval()

model = conf.RegressionJackknife(
    (
        preprocessing.StandardScaler() |
        linear_model.LinearRegression(intercept_lr=.1)
    ),
    confidence_level=0.9
)

validity = stats.Mean()
efficiency = stats.Mean()

for x, y in dataset:
    interval = model.predict_one(x, with_interval=True)
    validity.update(y in interval)
    efficiency.update(interval.width)
    model.learn_one(x, y)
```

The interval's validity is the proportion of times the true value is within the interval. We specified a confidence level of 90%, so we expect the validity to be around 90%.

```python
validity
```

```text
Mean: 0.939061
```

The interval's efficiency is the average width of the intervals.

```python
efficiency
```

```text
Mean: 4.078361
```

Lowering the confidence lowering will mechanically improve the efficiency.

## Methods

> [!NOTE]
>
> **learn\_one**
>
> Fits to a set of features `x` and a real-valued target `y`.
>
> **Parameters**
>
> - **x** — `dict[base.typing.FeatureName, object]`
> - **y** — `base.typing.RegTarget`
> - **kwargs** — `object`

> [!NOTE]
>
> **predict\_one**
>
> Predict the output of features `x`.
>
> **Parameters**
>
> - **x** — `dict[base.typing.FeatureName, object]`
> - **with\_interval** — `bool` — defaults to `False`
> - **kwargs** — `object`
>
> **Returns**
>
> `float | interval.Interval`: The prediction.

## References

- [Barber, Rina Foygel, Emmanuel J. Candes, Aaditya Ramdas, and Ryan J. Tibshirani. "Predictive inference with the jackknife+." The Annals of Statistics 49, no. 1 (2021): 486-507.](<https://www.stat.cmu.edu/~ryantibs/papers/jackknife.pdf>)
