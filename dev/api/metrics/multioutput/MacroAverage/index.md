# MacroAverage

Macro-average wrapper.

A copy of the provided metric is made for each output. The arithmetic average of all the metrics is returned.

## Parameters

- **metric**

  A classification or a regression metric.

## Attributes

- **bigger\_is\_better**

  Indicate if a high value is better than a low one or not.
- **metric**

  Gives access to the wrapped metric.
- **requires\_labels**
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
> - **y\_true**
> - **y\_pred**
> - **w** — defaults to `1.0`

> [!NOTE]
>
> **update**
>
> Update the metric.
>
> **Parameters**
>
> - **y\_true**
> - **y\_pred**
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
