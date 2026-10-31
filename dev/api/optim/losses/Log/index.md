# Log

Logarithmic loss.

This loss function expects each provided `y_pred` to be a logit. In other words if must be the raw output of a linear model or a neural network.

## Parameters

- **weight\_pos**

  *Type* → `float`

  *Default* → `1.0`
- **weight\_neg**

  *Type* → `float`

  *Default* → `1.0`

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

## References

- [Logit Wikipedia page](<https://en.wikipedia.org/wiki/Logit>)
