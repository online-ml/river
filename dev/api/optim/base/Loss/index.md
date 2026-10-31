# Loss

Base class for all loss functions.

## Methods

> [!NOTE]
>
> ****call****
>
> Returns the loss.
>
> **Parameters**
>
> - **y\_true** — `typing.Any`
> - **y\_pred** — `typing.Any`
>
> **Returns**
>
> `typing.Any`: The loss(es).

> [!NOTE]
>
> **gradient**
>
> Return the gradient with respect to y\_pred.
>
> **Parameters**
>
> - **y\_true** — `typing.Any`
> - **y\_pred** — `typing.Any`
>
> **Returns**
>
> `typing.Any`: The gradient(s).

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
