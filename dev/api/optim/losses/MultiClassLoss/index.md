# MultiClassLoss

A loss appropriate for multi-class classification tasks.

## Methods

> [!NOTE]
>
> ****call****
>
> Returns the loss.
>
> **Parameters**
>
> - **y\_true** — `base.typing.ClfTarget`
> - **y\_pred** — `dict[base.typing.ClfTarget, float]`
>
> **Returns**
>
> `float`: The loss(es).

> [!NOTE]
>
> **gradient**
>
> Return the gradient with respect to y\_pred.
>
> **Parameters**
>
> - **y\_true** — `base.typing.ClfTarget`
> - **y\_pred** — `dict[base.typing.ClfTarget, float]`
>
> **Returns**
>
> `dict&#91;base.typing.ClfTarget, float&#93;`: The gradient(s).

> [!NOTE]
>
> **mean\_func**
>
> Mean function.
>
> This is the inverse of the link function. Typically, a loss function takes as input the raw output of a model. In the case of classification, the raw output would be logits. The mean function can be used to convert the raw output into a value that makes sense to the user, such as a probability.
>
> **Parameters**
>
> - **y\_pred** — `typing.Any`
>
> **Returns**
>
> `typing.Any`: The adjusted prediction(s).
