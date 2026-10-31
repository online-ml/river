# Poisson

Poisson loss.

The Poisson loss is usually more suited for regression with count data than the squared loss.

Mathematically, it is defined as

\\\[L = exp(p\_i) - y\_i \\times p\_i\\\]

Its gradient w.r.t. to \\(p\_i\\) is

\\\[\\frac{\\partial L}{\\partial p\_i} = exp(p\_i) - y\_i\\\]

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
