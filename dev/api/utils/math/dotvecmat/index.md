# dotvecmat

Vector times matrix from left side, i.e. transpose(x)A.

## Parameters

- **x**

  *Type* → `Mapping[Any, float]`
- **A**

  *Type* → `Mapping[tuple[Any, Any], float]`

## Examples

```python
from river import utils

x = {0: 4, 1: 5}

A = {
    (0, 0): 0, (0, 1): 1,
    (1, 0): 2, (1, 1): 3
}

C = utils.math.dotvecmat(x, A)
print(C)
```

```text
{0: 10.0, 1: 19.0}
```
