# RegressionMetric

Mother class for all regression metrics.

## Attributes

- **bigger\_is\_better**

  Indicate if a high value is better than a low one or not.
- **works\_with\_weights**

  Indicate whether the model takes into consideration the effect of sample weights

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

> [!NOTE]
>
> **works\_with**
>
> Indicates whether or not a metric can work with a given model.
>
> **Parameters**
>
> - **model** — `base.Estimator`
