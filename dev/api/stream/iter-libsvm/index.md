# iter\_libsvm

Iterates over a dataset in LIBSVM format.

The LIBSVM format is a popular way in the machine learning community to store sparse datasets. Only numerical feature values are supported. The feature names will be considered as strings.

## Parameters

- **filepath\_or\_buffer**

  *Type* → `FilePath | typing.TextIO`

  Either a string indicating the location of a file, or a buffer object that has a `read` method.
- **target\_type**

  *Type* → `type[typing.Any]`

  *Default* → `<class 'float'>`

  The type of the target value.
- **compression**

  *Type* → `Compression | None`

  *Default* → `infer`

  For on-the-fly decompression of on-disk data. If this is set to 'infer' and `filepath_or_buffer` is a path, then the decompression method is inferred for the following extensions: '.gz', '.zip'.

## Examples

```python
import io
from river import stream

data = io.StringIO('''+1 x:-134.26 y:0.2563
1 x:-12 z:0.3
-1 y:.25
''')

for x, y in stream.iter_libsvm(data, target_type=int):
    print(y, x)
```

```text
1 {'x': -134.26, 'y': 0.2563}
1 {'x': -12.0, 'z': 0.3}
-1 {'y': 0.25}
```

## References

- [LIBSVM documentation](<https://www.csie.ntu.edu.tw/~cjlin/libsvm/>)
