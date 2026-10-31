# iter\_frame

Iterates over the rows of a dataframe.

This is a dataframe-agnostic iterator: it works with any eager dataframe supported by [Narwhals](<https://narwhals-dev.github.io/narwhals/>) (pandas, polars, PyArrow, Modin, cuDF, ...). It supersedes [`stream.iter_pandas`](<https://riverml.xyz/dev/api/stream/iter-pandas>) and [`stream.iter_polars`](<https://riverml.xyz/dev/api/stream/iter-polars>).

Rows are read directly from the dataframe via Narwhals, so each cell keeps its native per-column type (no conversion to a single `numpy` array, which would otherwise coerce a mixed-type frame to a common dtype).

Note that vaex is *not* supported here: Narwhals only exposes it through the dataframe interchange protocol, which cannot iterate rows without materializing the whole frame. Use `stream.iter_vaex` instead, which streams a vaex dataframe lazily.

## Parameters

- **X**

  *Type* → `IntoFrame`

  A dataframe of features. Any eager dataframe supported by Narwhals will work.
- **y**

  *Type* → `IntoSeries | IntoDataFrame | None`

  *Default* → `None`

  A series, or a dataframe with one column per target.
- **shuffle**

  *Type* → `bool`

  *Default* → `False`

  Whether to shuffle the rows before iterating over them. This materializes the whole stream in memory, as the order can only be permuted once every row is known.
- **seed**

  *Type* → `int | None`

  *Default* → `None`

  Random seed used for shuffling. Only used when `shuffle` is `True`.

## Examples

```python
from river import stream
```

The same code works regardless of the dataframe library. With pandas:

```python
import pandas as pd
X = pd.DataFrame({
    'x1': [1, 2, 3, 4],
    'x2': ['blue', 'yellow', 'yellow', 'blue'],
    'y': [True, False, False, True]
})
y = X.pop('y')

for xi, yi in stream.iter_frame(X, y):
    print(xi, yi)
```

```text
{'x1': 1, 'x2': 'blue'} True
{'x1': 2, 'x2': 'yellow'} False
{'x1': 3, 'x2': 'yellow'} False
{'x1': 4, 'x2': 'blue'} True
```

With polars:

```python
import polars as pl
X = pl.DataFrame({
    'x1': [1, 2, 3, 4],
    'x2': ['blue', 'yellow', 'yellow', 'blue'],
    'y': [True, False, False, True]
})
y = X.get_column('y')
X = X.drop('y')

for xi, yi in stream.iter_frame(X, y):
    print(xi, yi)
```

```text
{'x1': 1, 'x2': 'blue'} True
{'x1': 2, 'x2': 'yellow'} False
{'x1': 3, 'x2': 'yellow'} False
{'x1': 4, 'x2': 'blue'} True
```

And with PyArrow:

```python
import pyarrow as pa
X = pa.table({
    'x1': [1, 2, 3, 4],
    'x2': ['blue', 'yellow', 'yellow', 'blue'],
    'y': [True, False, False, True]
})
y = X.column('y')
X = X.drop(['y'])

for xi, yi in stream.iter_frame(X, y):
    print(xi, yi)
```

```text
{'x1': 1, 'x2': 'blue'} True
{'x1': 2, 'x2': 'yellow'} False
{'x1': 3, 'x2': 'yellow'} False
{'x1': 4, 'x2': 'blue'} True
```
