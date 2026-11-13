# matmul2d

Multiplication for 2D matrices.

## Parameters

- **A**

  *Type* → `Mapping[tuple[Any, Any], float]`
- **B**

  *Type* → `Mapping[tuple[Any, Any], float]`

## Examples

```python
import pprint
from river import utils

A = {
    (0, 0): 2, (0, 1): 0, (0, 2): 4,
    (1, 0): 5, (1, 1): 6, (1, 2): 0
}

B = {
    (0, 0): 1, (0, 1): 1, (0, 2): 0, (0, 3): 0,
    (1, 0): 2, (1, 1): 0, (1, 2): 1, (1, 3): 3,
    (2, 0): 4, (2, 1): 0, (2, 2): 0, (2, 3): 0
}

C = utils.math.matmul2d(A, B)
pprint.pprint(C)
```

```text
{(0, 0): 18.0,
    (0, 1): 2.0,
    (0, 2): 0.0,
    (0, 3): 0.0,
    (1, 0): 17.0,
    (1, 1): 5.0,
    (1, 2): 6.0,
    (1, 3): 18.0}
```
