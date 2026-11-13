# outer

Outer-product between two vectors.

## Parameters

- **u**

  *Type* → `Mapping[Any, float]`
- **v**

  *Type* → `Mapping[Any, float]`

## Examples

```python
import pprint
from river import utils

u = dict(enumerate((1, 2, 3)))
v = dict(enumerate((2, 4, 8)))

uTv = utils.math.outer(u, v)
pprint.pprint(uTv)
```

```text
{(0, 0): 2,
    (0, 1): 4,
    (0, 2): 8,
    (1, 0): 4,
    (1, 1): 8,
    (1, 2): 16,
    (2, 0): 6,
    (2, 1): 12,
    (2, 2): 24}
```
