# iter\_pandas

Iterates over the rows of a `pandas.DataFrame`.

.. deprecated:: Use [`stream.iter_frame`](<https://riverml.xyz/dev/api/stream/iter-frame>) instead, which works with any eager dataframe (pandas, polars, PyArrow, ...). [`stream.iter_pandas`](<https://riverml.xyz/dev/api/stream/iter-pandas>) will be removed in a future release.

## Parameters

- **X**

  *Type* → `pd.DataFrame`

  A dataframe of features.
- **y**

  *Type* → `pd.Series | pd.DataFrame | None`

  *Default* → `None`

  A series or a dataframe with one column per target.

## Examples

```python
import pandas as pd
from river import stream

X = pd.DataFrame({
    'x1': [1, 2, 3, 4],
    'x2': ['blue', 'yellow', 'yellow', 'blue'],
    'y': [True, False, False, True]
})
y = X.pop('y')

for xi, yi in stream.iter_pandas(X, y):
    print(xi, yi)
```

```text
{'x1': 1, 'x2': 'blue'} True
{'x1': 2, 'x2': 'yellow'} False
{'x1': 3, 'x2': 'yellow'} False
{'x1': 4, 'x2': 'blue'} True
```
