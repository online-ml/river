# AutoCorr

Measures the serial correlation.

This method computes the Pearson correlation between the current value and the value seen `n` steps before.

## Parameters

- **lag**

  *Type* → `int`

## Attributes

- **name**

## Examples

The following examples are taken from the [pandas documentation](<https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.autocorr.html>).

```python
from river import stats

auto_corr = stats.AutoCorr(lag=1)
for x in [0.25, 0.5, 0.2, -0.05]:
    auto_corr.update(x)
    print(auto_corr.get())
```

```text
0
0
-1.0
0.103552
```

```python
auto_corr = stats.AutoCorr(lag=2)
for x in [0.25, 0.5, 0.2, -0.05]:
    auto_corr.update(x)
    print(auto_corr.get())
```

```text
0
0
0
-1.0
```

```python
auto_corr = stats.AutoCorr(lag=1)
for x in [1, 0, 0, 0]:
    auto_corr.update(x)
    print(auto_corr.get())
```

```text
0
0
0
0
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
