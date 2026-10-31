# IQR

Computes the interquartile range.

## Parameters

- **q\_inf**

  *Type* → `float`

  *Default* → `0.25`

  Desired inferior quantile, must be between 0 and 1. Defaults to `0.25`.
- **q\_sup**

  *Type* → `float`

  *Default* → `0.75`

  Desired superior quantile, must be between 0 and 1. Defaults to `0.75`.

## Attributes

- **name**

## Examples

```python
from river import stats

iqr = stats.IQR(q_inf=0.25, q_sup=0.75)

for i in range(0, 1001):
    iqr.update(i)
    if i % 100 == 0:
        print(iqr.get())
```

```text
0.0
50.0
100.0
150.0
200.0
250.0
300.0
350.0
400.0
450.0
500.0
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
