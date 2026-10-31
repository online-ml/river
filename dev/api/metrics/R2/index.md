# R2

Coefficient of determination (\\(R^2\\)) score

The coefficient of determination, denoted \\(R^2\\) or \\(r^2\\), is the proportion of the variance in the dependent variable that is predictable from the independent variable(s). [^/dev/api/metrics/R2/#1]

Best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse). A constant model that always predicts the expected value of \\(y\\), disregarding the input features, would get a \\(R^2\\) score of 0.0.

\\(R^2\\) is not defined when less than 2 samples have been observed. This implementation returns 0.0 in this case.

## Attributes

- **bigger\_is\_better**

  Indicate if a high value is better than a low one or not.
- **works\_with\_weights**

  Indicate whether the model takes into consideration the effect of sample weights

## Examples

```python
from river import metrics

y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]

metric = metrics.R2()

for yt, yp in zip(y_true, y_pred):
    metric.update(yt, yp)
    print(metric.get())
```

```text
0.0
0.9183
0.9230
0.9486
```

## Methods

> [!NOTE]
>
> **get**
>
> Return the current value of the metric.

> [!NOTE]
>
> **is\_better\_than**
>
> Indicate if the current metric is better than another one.
>
> **Parameters**
>
> - **other**

> [!NOTE]
>
> **revert**
>
> Revert the metric.
>
> **Parameters**
>
> - **y\_true** — `float`
> - **y\_pred** — `float`
> - **w** — defaults to `1.0`

> [!NOTE]
>
> **update**
>
> Update the metric.
>
> **Parameters**
>
> - **y\_true** — `float`
> - **y\_pred** — `float`
> - **w** — defaults to `1.0`

> [!NOTE]
>
> **works\_with**
>
> Indicates whether or not a metric can work with a given model.
>
> **Parameters**
>
> - **model** — `base.Estimator`

[^/dev/api/metrics/R2/#1]: [Coefficient of determination (Wikipedia)](<https://en.wikipedia.org/wiki/Coefficient_of_determination>)
