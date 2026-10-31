# Hinge

Computes the hinge loss.

Mathematically, it is defined as

\\\[L = max(0, 1 - p\_i \* y\_i)\\\]

Its gradient w.r.t. to \\(p\_i\\) is

\\\[ \\\\frac{\\\\partial L}{\\\\partial y\_i} = \\\\left\\{ \\\\begin{array}{ll} \\\\ 0 &amp; p\_iy\_i \\geqslant 1 \\\\\\\\ \\\\ - y\_i &amp; p\_iy\_i \< 1 \\\\end{array} \\\\right. \\\]

## Parameters

- **threshold**

  *Type* → `float`

  *Default* → `1.0`

  Margin threshold. 1 yield the loss used in SVMs, whilst 0 is equivalent to the loss used in the Perceptron algorithm.

## Examples

```python
from river import optim

loss = optim.losses.Hinge(threshold=1)
loss(1, .2)
```

```text
0.8
```

```python
loss.gradient(1, .2)
```

```text
-1
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
