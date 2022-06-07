from river import base

from . import utils


def iter_libsvm(
    filepath_or_buffer: str, target_type=float, compression="infer"
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
    buffer = filepath_or_buffer
    should_close = False
    if not hasattr(buffer, "read"):
        should_close = False
        buffer = utils.open_filepath(buffer, compression)

    def split_pair(pair):
        name, value = pair.split(":")
        value = float(value)
        return name, value

    for line in buffer:

        # Remove carriage return and whitespace
        line = line.rstrip()
        # Remove potential end of line comments
        line = line.split("#")[0]

        y, x_str = line.split(" ", maxsplit=1)
        y = target_type(y)
        x = dict([split_pair(pair) for pair in x_str.split(" ")])
        yield x, y

    # Close the file if we opened it
    if should_close:
        buffer.close()  # type: ignore
