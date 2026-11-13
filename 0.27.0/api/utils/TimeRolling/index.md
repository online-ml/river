# TimeRolling

A generic wrapper for performing time rolling computations.

This can be wrapped around any class whose instances implement both an `update` and a `revert` method. The wrapped class is instantiated internally; any extra keyword arguments are forwarded to its constructor. Inputs to `update` are stored in a queue, and elements of the queue are popped when they are too old.

## Parameters

- **cls**

  *Type* → `type[_T] | _T`

  A class whose instances implement `update` and `revert`. Passing a pre-built instance is deprecated and will be removed in a future release.
- **period**

  *Type* → `dt.timedelta`

  A duration of time, expressed as a `datetime.timedelta`.

## Examples

For instance, here is how you can compute a rolling average over a period of 3 days:

```python
from river import stats, utils

X = {
    dt.datetime(2019, 1, 1): 1,
    dt.datetime(2019, 1, 2): 5,
    dt.datetime(2019, 1, 3): 9,
    dt.datetime(2019, 1, 4): 13
}

rmean = utils.TimeRolling(stats.Mean, period=dt.timedelta(days=3))
for t, x in X.items():
    rmean.update(x, t=t)
    print(rmean.get())
```

```text
1.0
3.0
5.0
9.0
```

## Methods

> [!NOTE]
>
> **update**
