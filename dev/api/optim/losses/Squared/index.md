# Squared

Squared loss, also known as the L2 loss.

Mathematically, it is defined as

\\\[L = (p\_i - y\_i) ^ 2\\\]

Its gradient w.r.t. to \\(p\_i\\) is

\\\[\\frac{\\partial L}{\\partial p\_i} = 2 (p\_i - y\_i)\\\]

One thing to note is that this convention is consistent with Vowpal Wabbit and PyTorch, but not with scikit-learn. Indeed, scikit-learn divides the loss by 2, making the 2 disappear in the gradient.

## Examples

```python
from river import optim

loss = optim.losses.Squared()
loss(-4, 5)
```

```text
81
```

```python
loss.gradient(-4, 5)
```

```text
18
```

```python
loss.gradient(5, -4)
```

```text
-18
```

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
