from __future__ import annotations

import os
import typing

from river import base
from river.stream import utils

if typing.TYPE_CHECKING:
    from river.base.typing import FeatureName
    from river.stream.typing import Compression, FilePath


def iter_libsvm(
    filepath_or_buffer: FilePath | typing.TextIO,
    target_type: type[typing.Any] = float,
    compression: Compression | None = "infer",
) -> base.typing.Stream:
    """Iterates over a dataset in LIBSVM format.

    The LIBSVM format is a popular way in the machine learning community to store sparse datasets.
    Only numerical feature values are supported. The feature names will be considered as strings.

    Parameters
    ----------
    filepath_or_buffer
        Either a string indicating the location of a file, or a buffer object that has a `read`
        method.
    target_type
        The type of the target value.
    compression
        For on-the-fly decompression of on-disk data. If this is set to 'infer' and
        `filepath_or_buffer` is a path, then the decompression method is inferred for the
        following extensions: '.gz', '.zip'.

    Examples
    --------

    >>> import io
    >>> from river import stream

    >>> data = io.StringIO('''+1 x:-134.26 y:0.2563
    ... 1 x:-12 z:0.3
    ... -1 y:.25
    ... ''')

    >>> for x, y in stream.iter_libsvm(data, target_type=int):
    ...     print(y, x)
    1 {'x': -134.26, 'y': 0.2563}
    1 {'x': -12.0, 'z': 0.3}
    -1 {'y': 0.25}

    References
    ----------
    [^1]: [LIBSVM documentation](https://www.csie.ntu.edu.tw/~cjlin/libsvm/)

    """

    # If a file is not opened, then we open it
    if isinstance(filepath_or_buffer, (str, os.PathLike)):
        buffer = utils.open_filepath(filepath_or_buffer, compression)
        should_close = True
    else:
        buffer = filepath_or_buffer
        should_close = False

    for line in buffer:
        # Remove carriage return and whitespace
        line = line.rstrip()
        # Remove potential end of line comments
        line = line.split("#")[0]

        y, x_str = line.split(" ", maxsplit=1)
        y = target_type(y)
        x = dict([_split_pair(pair) for pair in x_str.split(" ")])
        yield x, y

    # Close the file if we opened it
    if should_close:
        buffer.close()


def _split_pair(pair: str) -> tuple[FeatureName, float]:
    name, value = pair.split(":")
    return name, float(value)
