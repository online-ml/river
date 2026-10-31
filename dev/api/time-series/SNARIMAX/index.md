# SNARIMAX

SNARIMAX model.

SNARIMAX stands for (S)easonal (N)on-linear (A)uto(R)egressive (I)ntegrated (M)oving-(A)verage with e(X)ogenous inputs model.

This model generalizes many established time series models in a single interface that can be trained online. It assumes that the provided training data is ordered in time and is uniformly spaced. It is made up of the following components:

- S (Seasonal)
- N (Non-linear): Any online regression model can be used, not necessarily a linear regression

  as is done in textbooks. - AR (Autoregressive): Lags of the target variable are used as features.
- I (Integrated): The model can be fitted on a differenced version of a time series. In this

  context, integration is the reverse of differencing. - MA (Moving average): Lags of the errors are used as features.
- X (Exogenous): Users can provide additional features. Care has to be taken to include

  features that will be available both at training and prediction time.

Each of these components can be switched on and off by specifying the appropriate parameters. Classical time series models such as AR, MA, ARMA, and ARIMA can thus be seen as special parametrizations of the SNARIMAX model.

This model is tailored for time series that are homoskedastic. In other words, it might not work well if the variance of the time series varies widely along time.

## Parameters

- **p**

  *Type* → `int`

  Order of the autoregressive part. This is the number of past target values that will be included as features.
- **d**

  *Type* → `int`

  Differencing order.
- **q**

  *Type* → `int`

  Order of the moving average part. This is the number of past error terms that will be included as features.
- **m**

  *Type* → `int`

  *Default* → `1`

  Season length used for extracting seasonal features. If you believe your data has a seasonal pattern, then set this accordingly. For instance, if the data seems to exhibit a yearly seasonality, and that your data is spaced by month, then you should set this to 12. Note that for this parameter to have any impact you should also set at least one of the `p`, `d`, and `q` parameters.
- **sp**

  *Type* → `int`

  *Default* → `0`

  Seasonal order of the autoregressive part. This is the number of past target values that will be included as features.
- **sd**

  *Type* → `int`

  *Default* → `0`

  Seasonal differencing order.
- **sq**

  *Type* → `int`

  *Default* → `0`

  Seasonal order of the moving average part. This is the number of past error terms that will be included as features.
- **regressor**

  *Type* → `base.Regressor | compose.Pipeline | None`

  *Default* → `None`

  The online regression model to use. By default, a `preprocessing.StandardScaler` piped with a `linear_model.LinearRegression` will be used.

## Attributes

- **differencer** (`Differencer`)
- **y\_trues** (`collections.deque`)

  The `p` past target values.
- **errors** (`collections.deque`)

  The `q` past error values.

## Examples

```python
import datetime as dt
from river import datasets
from river import time_series
from river import utils

period = 12
model = time_series.SNARIMAX(
    p=period,
    d=1,
    q=period,
    m=period,
    sd=1
)

for t, (x, y) in enumerate(datasets.AirlinePassengers()):
    model.learn_one(y)

horizon = 12
future = [
    {'month': dt.date(year=1961, month=m, day=1)}
    for m in range(1, horizon + 1)
]
forecast = model.forecast(horizon=horizon)
for x, y_pred in zip(future, forecast):
    print(x['month'], f'{y_pred:.3f}')
```

```text
1961-01-01 494.542
1961-02-01 450.825
1961-03-01 484.972
1961-04-01 576.401
1961-05-01 559.489
1961-06-01 612.251
1961-07-01 722.410
1961-08-01 674.604
1961-09-01 575.716
1961-10-01 562.808
1961-11-01 477.049
1961-12-01 515.191
```

Classic ARIMA models learn solely on the time series values. You can also include features built at each step.

```python
import calendar
import math
from river import compose
from river import linear_model
from river import optim
from river import preprocessing

def get_month_distances(x):
    return {
        calendar.month_name[month]: math.exp(-(x['month'].month - month) ** 2)
        for month in range(1, 13)
    }

def get_ordinal_date(x):
    return {'ordinal_date': x['month'].toordinal()}

extract_features = compose.TransformerUnion(
    get_ordinal_date,
    get_month_distances
)

model = (
    extract_features |
    time_series.SNARIMAX(
        p=1,
        d=0,
        q=0,
        m=12,
        sp=3,
        sq=6,
        regressor=(
            preprocessing.StandardScaler() |
            linear_model.LinearRegression(
                intercept_init=110,
                optimizer=optim.SGD(0.01),
                intercept_lr=0.3
            )
        )
    )
)

for x, y in datasets.AirlinePassengers():
    model.learn_one(x, y)

forecast = model.forecast(horizon=horizon)
for x, y_pred in zip(future, forecast):
    print(x['month'], f'{y_pred:.3f}')
```

```text
1961-01-01 444.821
1961-02-01 432.612
1961-03-01 457.739
1961-04-01 465.544
1961-05-01 476.575
1961-06-01 516.255
1961-07-01 565.405
1961-08-01 572.470
1961-09-01 512.645
1961-10-01 475.919
1961-11-01 438.033
1961-12-01 456.892
```

## Methods

> [!NOTE]
>
> **forecast**
>
> Makes forecast at each step of the given horizon.
>
> **Parameters**
>
> - **horizon** — `int`
> - **xs** — `list[dict] | None` — defaults to `None`

> [!NOTE]
>
> **learn\_one**
>
> Updates the model.
>
> **Parameters**
>
> - **y** — `float`
> - **x** — `dict | None` — defaults to `None`

## References

- [ARMA - Wikipedia](<https://en.wikipedia.org/wiki/Autoregressive_moving-average_model>)
- [NARX - Wikipedia](<https://en.wikipedia.org/wiki/Nonlinear_autoregressive_exogenous_model>)
- [ARIMA - Forecasting: Principles and Practice](<https://otexts.com/fpp2/arima.html>)
- [Anava, O., Hazan, E., Mannor, S. and Shamir, O., 2013, June. Online learning for time series prediction. In Conference on learning theory (pp. 172-184)](<https://arxiv.org/pdf/1302.6927.pdf>)
