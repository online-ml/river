from __future__ import annotations

import csv
import datetime as dt
import os
import random
import typing

from river import base
from river.stream import utils

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from river.base.typing import FeatureName
    from river.stream.typing import Compression, FilePath

__all__ = ["iter_csv"]


class DictReader(csv.DictReader):
    """Overlay on top of `csv.DictReader` which allows sampling."""

    def __init__(
        self, fraction: float, rng: random.Random, *args: typing.Any, **kwargs: typing.Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.fraction = fraction
        self.rng = rng
        # NOTE: `fieldnames` reads the header row, and is `None` for an empty file,
        # in which case the first `next(self.reader)` below stops the iteration anyway.
        self.names: Sequence[FeatureName] = self.fieldnames or ()

    def __next__(self) -> dict[FeatureName, typing.Any]:
        row = next(self.reader)

        if self.fraction < 1:
            while self.rng.random() > self.fraction:
                row = next(self.reader)

        return dict(zip(self.names, row))


def iter_csv(
    filepath_or_buffer: FilePath | typing.TextIO,
    target: str | list[str] | None = None,
    converters: dict[FeatureName, Callable[[typing.Any], typing.Any]] | None = None,
    parse_dates: dict[FeatureName, str] | None = None,
    drop: list[str] | None = None,
    drop_nones: bool = False,
    fraction: float = 1.0,
    compression: Compression | None = "infer",
    seed: int | None = None,
    field_size_limit: int | None = None,
    **kwargs: typing.Any,
) -> base.typing.Stream:
    """Iterates over rows from a CSV file.

    Reading CSV files can be quite slow. If, for whatever reason, you're going to loop through
    the same file multiple times, then we recommend that you to use the `stream.Cache` utility.

    Parameters
    ----------
    filepath_or_buffer
        Either a string indicating the location of a file, or a buffer object that has a
        `read` method.
    target
        A single target column is assumed if a string is passed. A multiple output scenario
        is assumed if a list of strings is passed. A `None` value will be assigned to each `y`
        if this parameter is omitted.
    converters
        All values in the CSV are interpreted as strings by default. You can use this parameter to
        cast values to the desired type. This should be a `dict` mapping feature names to callables
        used to parse their associated values. Note that a callable may be a type, such as `float`
        and `int`.
    parse_dates
        A `dict` mapping feature names to a format passed to the `datetime.datetime.strptime`
        method.
    drop
        Fields to ignore.
    drop_nones
        Whether or not to drop fields where the value is a `None`.
    fraction
        Sampling fraction.
    compression
        For on-the-fly decompression of on-disk data. If this is set to 'infer' and
        `filepath_or_buffer` is a path, then the decompression method is inferred for the
        following extensions: '.gz', '.zip'.
    seed
        If specified, the sampling will be deterministic.
    field_size_limit
        If not `None`, this will be passed to the `csv.field_size_limit` function.
    kwargs
        All other keyword arguments are passed to the underlying `csv.DictReader`.

    Returns
    -------

    By default each feature value will be of type `str`. You can use the `converters` and
    `parse_dates` parameters to convert them as you see fit.

    Examples
    --------

    Although this function is designed to handle different kinds of inputs, the most common
    use case is to read a file on the disk. We'll first create a little CSV file to illustrate.

    >>> tv_shows = '''name,year,rating
    ... Planet Earth II,2016,9.5
    ... Planet Earth,2006,9.4
    ... Band of Brothers,2001,9.4
    ... Breaking Bad,2008,9.4
    ... Chernobyl,2019,9.4
    ... '''
    >>> with open('tv_shows.csv', mode='w') as f:
    ...     _ = f.write(tv_shows)

    We can now go through the rows one by one. We can use the `converters` parameter to cast
    the `rating` field value as a `float`. We can also convert the `year` to a `datetime` via
    the `parse_dates` parameter.

    >>> from river import stream

    >>> params = {
    ...     'converters': {'rating': float},
    ...     'parse_dates': {'year': '%Y'}
    ... }
    >>> for x, y in stream.iter_csv('tv_shows.csv', **params):
    ...     print(x, y)
    {'name': 'Planet Earth II', 'year': datetime.datetime(2016, 1, 1, 0, 0), 'rating': 9.5} None
    {'name': 'Planet Earth', 'year': datetime.datetime(2006, 1, 1, 0, 0), 'rating': 9.4} None
    {'name': 'Band of Brothers', 'year': datetime.datetime(2001, 1, 1, 0, 0), 'rating': 9.4} None
    {'name': 'Breaking Bad', 'year': datetime.datetime(2008, 1, 1, 0, 0), 'rating': 9.4} None
    {'name': 'Chernobyl', 'year': datetime.datetime(2019, 1, 1, 0, 0), 'rating': 9.4} None

    The value of `y` is always `None` because we haven't provided a value for the `target`
    parameter. Here is an example where a `target` is provided:

    >>> dataset = stream.iter_csv('tv_shows.csv', target='rating', **params)
    >>> for x, y in dataset:
    ...     print(x, y)
    {'name': 'Planet Earth II', 'year': datetime.datetime(2016, 1, 1, 0, 0)} 9.5
    {'name': 'Planet Earth', 'year': datetime.datetime(2006, 1, 1, 0, 0)} 9.4
    {'name': 'Band of Brothers', 'year': datetime.datetime(2001, 1, 1, 0, 0)} 9.4
    {'name': 'Breaking Bad', 'year': datetime.datetime(2008, 1, 1, 0, 0)} 9.4
    {'name': 'Chernobyl', 'year': datetime.datetime(2019, 1, 1, 0, 0)} 9.4

    Finally, let's delete the example file.

    >>> import os; os.remove('tv_shows.csv')

    """

    # Set the field size limit
    limit = csv.field_size_limit()
    if field_size_limit is not None:
        csv.field_size_limit(field_size_limit)

    # If a file is not opened, then we open it
    if isinstance(filepath_or_buffer, (str, os.PathLike)):
        buffer = utils.open_filepath(filepath_or_buffer, compression)
        should_close = True
    else:
        buffer = filepath_or_buffer
        should_close = False

    for x in DictReader(fraction=fraction, rng=random.Random(seed), f=buffer, **kwargs):
        if drop:
            for dropped in drop:
                del x[dropped]

        # Cast the values to the given types
        if converters is not None:
            for name, convert in converters.items():
                x[name] = convert(x[name])

        # Drop Nones
        if drop_nones:
            for name in list(x):
                if x[name] is None:
                    del x[name]

        # Parse the dates
        if parse_dates is not None:
            for name, fmt in parse_dates.items():
                x[name] = dt.datetime.strptime(x[name], fmt)

        # Separate the target from the features
        y = None
        if isinstance(target, list):
            y = {name: x.pop(name) for name in target}
        elif target is not None:
            y = x.pop(target)

        yield x, y

    # Close the file if we opened it
    if should_close:
        buffer.close()

    # Reset the file size limit to it's original value
    csv.field_size_limit(limit)
