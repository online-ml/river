from __future__ import annotations

import scipy.io.arff
from scipy.io.arff._arffread import read_header

from river import base

from . import utils


def iter_arff(
    filepath_or_buffer, target: str | None = None, compression="infer"
) -> base.typing.Stream:
    """Iterates over rows from an ARFF file.

    Parameters
    ----------
    filepath_or_buffer
        Either a string indicating the location of a file, or a buffer object that has a
        `read` method.
    target
        Name of the target field.
    compression
        For on-the-fly decompression of on-disk data. If this is set to 'infer' and
        `filepath_or_buffer` is a path, then the decompression method is inferred for the
        following extensions: '.gz', '.zip'.

    Examples
    --------

    >>> cars = '''
    ... @relation CarData
    ... @attribute make {Toyota, Honda, Ford, Chevrolet}
    ... @attribute model string
    ... @attribute year numeric
    ... @attribute price numeric
    ... @attribute mpg numeric
    ... @data
    ... Toyota, Corolla, 2018, 15000, 30.5
    ... Honda, Civic, 2019, 16000, 32.2
    ... Ford, Mustang, 2020, 25000, 25.0
    ... Chevrolet, Malibu, 2017, 18000, 28.9
    ... Toyota, Camry, 2019, 22000, 29.8
    ... '''
    >>> with open('cars.arff', mode='w') as f:
    ...     _ = f.write(cars)

    >>> from river import stream

    >>> for x, y in stream.iter_arff('cars.arff', target='price'):
    ...     print(x, y)
    {'make': 'Toyota', 'model': ' Corolla', 'year': 2018.0, 'mpg': 30.5} 15000.0
    {'make': 'Honda', 'model': ' Civic', 'year': 2019.0, 'mpg': 32.2} 16000.0
    {'make': 'Ford', 'model': ' Mustang', 'year': 2020.0, 'mpg': 25.0} 25000.0
    {'make': 'Chevrolet', 'model': ' Malibu', 'year': 2017.0, 'mpg': 28.9} 18000.0
    {'make': 'Toyota', 'model': ' Camry', 'year': 2019.0, 'mpg': 29.8} 22000.0

    Finally, let's delete the example file.

    >>> import os; os.remove('cars.arff')

    """

    # If a file is not opened, then we open it
    buffer = filepath_or_buffer
    if not hasattr(buffer, "read"):
        buffer = utils.open_filepath(buffer, compression)

    try:
        rel, attrs = read_header(buffer)
    except ValueError as e:
        msg = f"Error while parsing header, error was: {e}"
        raise scipy.io.arff.ParseArffError(msg)

    names = [attr.name for attr in attrs]
    # HACK
    casts = [float if attr.__class__.__name__ == "NumericAttribute" else None for attr in attrs]

    for r in buffer:
        if len(r) == 0:
            continue
        x = {
            name: cast(val) if cast else val
            for name, cast, val in zip(names, casts, r.rstrip().split(","))
        }
        try:
            y = x.pop(target) if target else None
        except KeyError as e:
            print(r)
            raise e

        yield x, y

    # Close the file if we opened it
    if buffer is not filepath_or_buffer:
        buffer.close()
