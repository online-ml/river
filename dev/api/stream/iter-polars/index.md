# iter\_polars

Iterates over the rows of a `polars.DataFrame`.

.. deprecated:: Use [`stream.iter_frame`](<https://riverml.xyz/dev/api/stream/iter-frame>) instead, which works with any eager dataframe (pandas, polars, PyArrow, ...). [`stream.iter_polars`](<https://riverml.xyz/dev/api/stream/iter-polars>) will be removed in a future release.

## Parameters

- **X**

  *Type* → `pl.DataFrame`

  A dataframe of features.
- **y**

  *Type* → `pl.Series | pl.DataFrame | None`

  *Default* → `None`

  A series or a dataframe with one column per target.

## Examples

```python
import polars as pl
from river import stream

X = pl.DataFrame({
    'x1': [1, 2, 3, 4],
    'x2': ['blue', 'yellow', 'yellow', 'blue'],
    'y': [True, False, False, True]
})
y = X.get_column('y')
X = X.drop('y')

for xi, yi in stream.iter_polars(X, y):
    print(xi, yi)
```

```text
{'x1': 1, 'x2': 'blue'} True
{'x1': 2, 'x2': 'yellow'} False
{'x1': 3, 'x2': 'yellow'} False
{'x1': 4, 'x2': 'blue'} True
```
