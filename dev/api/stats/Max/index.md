# Max

Running max.

## Attributes

- **max** (`float`)

  The current max.

## Examples

```python
from river import stats

X = [1, -4, 3, -2, 5, -6]
maximum = stats.Max()
for x in X:
    maximum.update(x)
    print(maximum.get())
```

```text
1
1
3
3
5
5
```

## Methods

> [!NOTE]
>
> **get**
>
> Return the current value of the statistic.

> [!NOTE]
>
> **update**
>
> Update the called instance.
>
> **Parameters**
>
> - **x** — `float`
