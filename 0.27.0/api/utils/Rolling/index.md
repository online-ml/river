# Rolling

A generic wrapper for performing rolling computations.

This can be wrapped around any class whose instances implement both an `update` and a `revert` method. The wrapped class is instantiated internally; any extra keyword arguments are forwarded to its constructor. Inputs to `update` are stored in a queue, and elements of the queue are popped when the window is full.

## Parameters

- **cls**

  *Type* → `type[_T] | _T`

  A class whose instances implement `update` and `revert`. Passing a pre-built instance is deprecated and will be removed in a future release.
- **window\_size**

  *Type* → `int`

  Size of the window.

## Attributes

- **window\_size**

## Examples

For instance, here is how you can compute a rolling average over a window of size 3:

```python
from river import stats, utils

X = [1, 3, 5, 7]
rmean = utils.Rolling(stats.Mean, window_size=3)

for x in X:
    rmean.update(x)
    print(rmean.get())
```

```text
1.0
2.0
3.0
5.0
```

Constructor arguments for the wrapped class are passed as keyword arguments:

```python
rvar = utils.Rolling(stats.Var, window_size=3, ddof=0)
```

## Methods

> [!NOTE]
>
> **update**
