# Newton

Online Newton Step (ONS) optimizer.

This optimizer uses second-order information (i.e. the Hessian of the cost function) in addition to first-order information (i.e. the gradient of the cost function). It maintains the matrix

\\\[A\_t = \\epsilon I + \\sum\_{s=1}^{t} g\_s g\_s^\\intercal\\\]

where \\(g\_s\\) is the gradient at step \\(s\\), and applies the update

\\\[w\_{t+1} = w\_t - \\eta A\_t^{-1} g\_t.\\\]

The inverse \\(A\_t^{-1}\\) is never computed from scratch. Instead it is updated in-place at each step with the Sherman-Morrison formula (see [`utils.math.sherman_morrison`](<https://riverml.xyz/dev/api/utils/math/sherman-morrison>)), which turns the rank-1 update \\(A\_t = A\_{t-1} + g\_t g\_t^\\intercal\\) into a single rank-1 update of the inverse. This costs \\(O(d^2)\\) time and memory, where \\(d\\) is the number of features seen so far.

## Parameters

- **lr** *(mutable)*

  *Type* → `int | float | optim.base.Scheduler`

  *Default* → `0.1`

  Learning rate. This corresponds to \\(1 / \\gamma\\) in the paper.
- **eps**

  *Type* → `float`

  *Default* → `1e-05`

  Regularization term used to initialize the Hessian to \\(\\epsilon I\\). Its inverse \\(1 / \\epsilon\\) is used as the starting value of \\(A\_t^{-1}\\), so a smaller `eps` allows larger steps early on.

## Attributes

- **learning\_rate**

## Examples

```python
from river import datasets
from river import evaluate
from river import linear_model
from river import metrics
from river import optim
from river import preprocessing

dataset = datasets.Phishing()
optimizer = optim.Newton()
model = (
    preprocessing.StandardScaler() |
    linear_model.LogisticRegression(optimizer)
)
metric = metrics.F1()

evaluate.progressive_val_score(dataset, model, metric)
```

```text
F1: 73.62%
```

## Methods

> [!NOTE]
>
> **look\_ahead**
>
> Updates a weight vector before a prediction is made.
>
> Parameters: w (dict): A dictionary of weight parameters. The weights are modified in-place. Returns: The updated weights.
>
> **Parameters**
>
> - **w** — `DictLike | VectorLike`

> [!NOTE]
>
> **step**
>
> Updates a weight vector given a gradient.
>
> **Parameters**
>
> - **w** — `DictLike | VectorLike`
> - **g** — `DictLike | VectorLike`
>
> **Returns**
>
> `DictLike | VectorLike`: The updated weights.

## References

- [Hazan, E., Agarwal, A. and Kale, S., 2007. Logarithmic regret algorithms for online convex optimization. Machine Learning, 69(2-3), pp.169-192](<https://www.cs.princeton.edu/~ehazan/papers/log-journal.pdf>)
- [Hazan, E., 2016. Introduction to online convex optimization. Foundations and Trends in Optimization, 2(3-4), pp.157-325 (Chapter 4 states the ONS regret theorem)](<https://arxiv.org/abs/1909.05207>)
- [Hazan, E., Koren, T. and Levy, K.Y., 2014. Logistic regression: Tight bounds for stochastic and online optimization. COLT 2014 — shows the exp-concavity constant of the logistic loss is e^{-D}, so ONS has no asymptotic edge over first-order methods on logistic regression until T is exponentially large](<https://proceedings.mlr.press/v35/hazan14a.html>)
