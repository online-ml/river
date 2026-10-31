# dot

Returns the dot product of two vectors represented as dicts.

## Parameters

- **x**

  *Type* → `Mapping[Any, float]`
- **y**

  *Type* → `Mapping[Any, float]`

## Examples

```python
from river import utils

x = {'x0': 1, 'x1': 2}
y = {'x1': 21, 'x2': 3}

utils.math.dot(x, y)
```

```text
42
```
