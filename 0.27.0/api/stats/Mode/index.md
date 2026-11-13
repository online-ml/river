# Mode

Running mode.

The mode is simply the most common value. An approximate mode can be computed by setting the number of first unique values to count.

## Parameters

- **k**

  *Type* → `int`

  *Default* → `25`

  Only the first `k` unique values will be included. If `k` equals -1, the exact mode is computed.

## Attributes

- **name**

## Examples

```python
from river import stats

X = ['sunny', 'cloudy', 'cloudy', 'rainy', 'rainy', 'rainy']
mode = stats.Mode(k=2)
for x in X:
    mode.update(x)
    print(mode.get())
```

```text
sunny
sunny
cloudy
cloudy
cloudy
cloudy
```

```python
mode = stats.Mode(k=-1)
for x in X:
    mode.update(x)
    print(mode.get())
```

```text
sunny
sunny
cloudy
cloudy
cloudy
rainy
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
> - **x** — `Hashable`
