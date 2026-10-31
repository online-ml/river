# iter\_array

Iterates over the rows from an array of features and an array of targets.

This method is intended to work with `numpy` arrays, but should also work with Python lists.

## Parameters

- **X**

  *Type* → `np.ndarray | list[_RowT]`

  A 2D array of features. This can also be a 1D array of strings, which can be the case if you're working with text.
- **y**

  *Type* → `np.ndarray | list[typing.Any] | None`

  *Default* → `None`

  An optional array of targets.
- **feature\_names**

  *Type* → `Sequence[base.typing.FeatureName] | None`

  *Default* → `None`

  An optional list of feature names. The features will be labeled with integers if no names are provided.
- **target\_names**

  *Type* → `Sequence[base.typing.FeatureName] | None`

  *Default* → `None`

  An optional list of output names. The outputs will be labeled with integers if no names are provided. Only applies if there are multiple outputs, i.e. if `y` is a 2D array.
- **shuffle**

  *Type* → `bool`

  *Default* → `False`

  Indicates whether or not to shuffle the input arrays before iterating over them.
- **seed**

  *Type* → `int | None`

  *Default* → `None`

  Random seed used for shuffling the data.
- **chunk\_size**

  *Type* → `int`

  *Default* → `1024`

  Number of rows converted at a time to native Python values. Only applies to 1D `numpy` arrays, i.e. texts and single-output targets. A larger value uses more memory while iterating.

## Examples

```python
from river import stream
import numpy as np

X = np.array([[1, 2, 3], [11, 12, 13]])
Y = np.array([True, False])

dataset = stream.iter_array(
    X, Y,
    feature_names=['x1', 'x2', 'x3']
)
for x, y in dataset:
    print(x, y)
```

```text
{'x1': 1, 'x2': 2, 'x3': 3} True
{'x1': 11, 'x2': 12, 'x3': 13} False
```

This also works with an array of texts:

```python
X = ["foo", "bar"]
dataset = stream.iter_array(X, Y)
for x, y in dataset:
    print(x, y)
```

```text
foo True
bar False
```
