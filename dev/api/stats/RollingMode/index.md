# RollingMode

Running mode over a window.

The mode is the most common value.

## Parameters

- **window\_size**

  *Type* → `int`

  Size of the rolling window.

## Attributes

- **counts** (`collections.defaultdict`)

  Value counts.

## Examples

```python
from river import stats

X = ['sunny', 'sunny', 'sunny', 'rainy', 'rainy', 'rainy', 'rainy']
rolling_mode = stats.RollingMode(window_size=2)
for x in X:
    rolling_mode.update(x)
    print(rolling_mode.get())
```

```text
sunny
sunny
sunny
sunny
rainy
rainy
rainy
```

```python
rolling_mode = stats.RollingMode(window_size=5)
for x in X:
    rolling_mode.update(x)
    print(rolling_mode.get())
```

```text
sunny
sunny
sunny
sunny
sunny
rainy
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
