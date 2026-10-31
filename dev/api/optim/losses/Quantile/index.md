# Quantile

Quantile loss.

## Parameters

- **alpha**

  *Type* → `float`

  *Default* → `0.5`

  Desired quantile to attain.

## Examples

```python
from river import optim

loss = optim.losses.Quantile(0.5)
loss(1, 3)
```

```text
1.0
```

```python
loss.gradient(1, 3)
```

```text
0.5
```

```python
loss.gradient(3, 1)
```

```text
-0.5
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

## References

- [Wikipedia article on quantile regression](<https://en.wikipedia.org/wiki/Quantile_regression>)
- [Derivative from WolframAlpha](<https://www.wolframalpha.com/input/?i=derivative+(y+-+p)+*+(alpha+-+Boole(y+-+p))+wrt+p>)
