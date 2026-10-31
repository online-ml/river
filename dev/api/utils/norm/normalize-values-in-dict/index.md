# normalize\_values\_in\_dict

Normalize the values in a dictionary using the given factor.

For each element in the dictionary, applies `value/factor`.

## Parameters

- **dictionary**

  *Type* → `T`

  Dictionary to normalize.
- **factor**

  *Type* → `float | None`

  *Default* → `None`

  Normalization factor value. If not set, use the sum of values.
- **inplace**

  *Type* → `bool`

  *Default* → `True`

  If True, perform operation in-place
- **raise\_error**

  *Type* → `bool`

  *Default* → `False`

  In case the normalization factor is either `0` or `None`: - `True`: raise an error. - `False`: return gracefully (if `inplace=False`, a copy of) `dictionary`.
