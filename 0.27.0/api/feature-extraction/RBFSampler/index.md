# RBFSampler

Extracts random features which approximate an RBF kernel.

This is a powerful way to give non-linear capacity to linear classifiers. This method is also called "random Fourier features" in the literature.

## Parameters

- **gamma**

  *Default* → `1.0`

  RBF kernel parameter in `(-gamma * x^2)`.
- **n\_components**

  *Default* → `100`

  Number of samples per original feature. Equals the dimensionality of the computed feature space.
- **seed**

  *Type* → `int | None`

  *Default* → `None`

  Random number seed.

## Examples

```python
from river import feature_extraction as fx
from river import linear_model as lm
from river import optim
from river import stream

X = [[0, 0], [1, 1], [1, 0], [0, 1]]
Y = [0, 0, 1, 1]

model = lm.LogisticRegression(optimizer=optim.SGD(.1))

for x, y in stream.iter_array(X, Y):
    model.learn_one(x, y)
    y_pred = model.predict_one(x)
    print(y, int(y_pred))
```

```text
0 0
0 0
1 0
1 1
```

```python
model = (
    fx.RBFSampler(seed=3) |
    lm.LogisticRegression(optimizer=optim.SGD(.1))
)

for x, y in stream.iter_array(X, Y):
    model.learn_one(x, y)
    y_pred = model.predict_one(x)
    print(y, int(y_pred))
```

```text
0 0
0 0
1 1
1 1
```

## Methods

> [!NOTE]
>
> **learn\_one**
>
> Update with a set of features `x`.
>
> A lot of transformers don't actually have to do anything during the `learn_one` step because they are stateless. For this reason the default behavior of this function is to do nothing. Transformers that however do something during the `learn_one` can override this method.
>
> **Parameters**
>
> - **x** — `dict[base.typing.FeatureName, Any]`

> [!NOTE]
>
> **transform\_one**
>
> Transform a set of features `x`.
>
> **Parameters**
>
> - **x** — `dict[base.typing.FeatureName, Any]`
> - **y** — defaults to `None`
>
> **Returns**
>
> `dict&#91;base.typing.FeatureName, Any&#93;`: The transformed values.

## References

- [Rahimi, A. and Recht, B., 2008. Random features for large-scale kernel machines. In Advances in neural information processing systems (pp. 1177-1184](<https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf>)
