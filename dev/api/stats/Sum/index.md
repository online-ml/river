# Sum

Running sum.

## Attributes

- **sum** (`float`)

  The running sum.

## Examples

```python
from river import stats

X = [-5, -3, -1, 1, 3, 5]
mean = stats.Sum()
for x in X:
    mean.update(x)
    print(mean.get())
```

```text
-5.0
-8.0
-9.0
-8.0
-5.0
0.0
```

```python
from river import utils

X = [1, -4, 3, -2, 2, 1]
rolling_sum = utils.Rolling(stats.Sum, window_size=2)
for x in X:
    rolling_sum.update(x)
    print(rolling_sum.get())
```

```text
1.0
-3.0
-1.0
1.0
0.0
3.0
```

## Methods

> [!NOTE]
>
> **get**
>
> Return the current value of the statistic.

> [!NOTE]
>
> **revert**

> [!NOTE]
>
> **update**
>
> Update the called instance.
>
> **Parameters**
>
> - **x** — `float`
