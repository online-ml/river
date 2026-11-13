# AbsMax

Running absolute max.

## Attributes

- **abs\_max** (`float`)

  The current absolute max.

## Examples

```python
from river import stats

X = [1, -4, 3, -2, 5, -6]
abs_max = stats.AbsMax()
for x in X:
    abs_max.update(x)
    print(abs_max.get())
```

```text
1
4
4
4
5
6
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
