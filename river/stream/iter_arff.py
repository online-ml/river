from __future__ import annotations

import functools
import os
import re
import typing

import scipy.io.arff
from scipy.io.arff._arffread import read_header

from river import base
from river.stream import utils

if typing.TYPE_CHECKING:
    from river.base.typing import FeatureName
    from river.stream.typing import Compression, FilePath


_ESCAPE_PATTERN = re.compile(r"\\([0-9]{1,3}|u[0-9a-f]{4}|.)")
_DENSE_VALUE_PATTERN = re.compile(
    r"""\s*('(?:\\.|[^'\\])*'|"(?:\\.|[^"\\])*"|[^\s,'"%{}]+|\{[^{}]*\})\s*(,|(?=%|$))"""
)
_DENSE_UNQUOTED_PATTERN = re.compile(r"""\s*[^\s,'"%{}]+\s*(?:,\s*[^\s,'"%{}]+\s*)*""")
_ESCAPES = {
    "\\": "\\",
    '"': '"',
    "'": "'",
    "t": "\t",
    "n": "\n",
    "r": "\r",
    "b": "\b",
    "f": "\f",
    "%": "%",
}


def _replace_escape(match: re.Match[str]) -> str:
    code = match[1]
    if code.isdecimal():
        return chr(int(code, 8))
    if code.startswith("u"):
        return chr(int(code[1:], 16))
    try:
        return _ESCAPES[code]
    except KeyError as error:
        raise ValueError from error


@functools.lru_cache(maxsize=256)
def _parse_quoted_value(value: str) -> str:
    quote = value[0]
    if value[-1] != quote:
        raise ValueError
    escaped = False
    for char in value[1:-1]:
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == quote:
            raise ValueError
    if escaped:
        raise ValueError
    return _ESCAPE_PATTERN.sub(_replace_escape, value[1:-1])


def _parse_dense_row(row: str, n_attrs: int) -> list[str]:
    if row.count(",") == n_attrs - 1 and "%" not in row and "{" not in row and "}" not in row:
        values = row.split(",")
        if " " not in row and "\t" not in row:
            return values
        return [value.strip() for value in values]
    if _DENSE_UNQUOTED_PATTERN.fullmatch(row):
        values = [value.strip() for value in row.split(",")]
        if len(values) != n_attrs:
            raise ValueError
        return values
    values = []
    position = 0
    while True:
        match = _DENSE_VALUE_PATTERN.match(row, position)
        if match is None:
            raise ValueError
        values.append(match[1])
        position = match.end()
        if match[2] != ",":
            break
    if len(values) == n_attrs + 1 and values[-1].startswith("{"):
        float(values.pop()[1:-1])
    if len(values) != n_attrs or any(value.startswith("{") for value in values):
        raise ValueError
    return values


def iter_arff(
    filepath_or_buffer: FilePath | typing.TextIO,
    target: str | list[str] | None = None,
    compression: Compression | None = "infer",
    sparse: bool = False,
    encoding: str = "utf-8",
) -> base.typing.Stream:
    """Iterate over rows from an ARFF file.

    Parameters
    ----------
    filepath_or_buffer
        Either a string indicating the location of a file, or a buffer object that has a
        `read` method.
    target
        Name(s) of the target field. If `None`, then the target field is ignored. If a list of
        names is passed, then a dictionary is returned instead of a single value.
    compression
        For on-the-fly decompression of on-disk data. If this is set to 'infer' and
        `filepath_or_buffer` is a path, then the decompression method is inferred for the
        following extensions: '.gz', '.zip'.
    sparse
        Whether the data is sparse or not.
    encoding
        Encoding used to open file paths. Defaults to UTF-8. Ignored when a text buffer is supplied.

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
    {'make': 'Toyota', 'model': 'Corolla', 'year': 2018.0, 'mpg': 30.5} 15000.0
    {'make': 'Honda', 'model': 'Civic', 'year': 2019.0, 'mpg': 32.2} 16000.0
    {'make': 'Ford', 'model': 'Mustang', 'year': 2020.0, 'mpg': 25.0} 25000.0
    {'make': 'Chevrolet', 'model': 'Malibu', 'year': 2017.0, 'mpg': 28.9} 18000.0
    {'make': 'Toyota', 'model': 'Camry', 'year': 2019.0, 'mpg': 29.8} 22000.0

    Finally, let's delete the example file.

    >>> import os; os.remove('cars.arff')

    ARFF files support sparse data. Let's create a sparse ARFF file.

    >>> sparse = '''
    ... % traindata
    ... @RELATION "traindata: -C 6"
    ... @ATTRIBUTE y0 {0, 1}
    ... @ATTRIBUTE y1 {0, 1}
    ... @ATTRIBUTE y2 {0, 1}
    ... @ATTRIBUTE y3 {0, 1}
    ... @ATTRIBUTE y4 {0, 1}
    ... @ATTRIBUTE y5 {0, 1}
    ... @ATTRIBUTE X0 NUMERIC
    ... @ATTRIBUTE X1 NUMERIC
    ... @ATTRIBUTE X2 NUMERIC
    ... @DATA
    ... { 3 1,6 0.863382,8 0.820094 }
    ... { 2 1,6 0.659761 }
    ... { 0 1,3 1,6 0.437881,8 0.818882 }
    ... { 2 1,6 0.676477,7 0.724635,8 0.755123 }
    ... '''

    >>> with open('sparse.arff', mode='w') as f:
    ...     _ = f.write(sparse)

    In addition, we'll specify that there are several target fields.

    >>> arff_stream = stream.iter_arff(
    ...     'sparse.arff',
    ...     target=['y0', 'y1', 'y2', 'y3', 'y4', 'y5'],
    ...     sparse=True
    ... )

    >>> for x, y in arff_stream:
    ...     print(x)
    ...     print(y)
    {'X0': 0.863382, 'X2': 0.820094}
    {'y0': '0', 'y1': '0', 'y2': '0', 'y3': '1', 'y4': '0', 'y5': '0'}
    {'X0': 0.659761}
    {'y0': '0', 'y1': '0', 'y2': '1', 'y3': '0', 'y4': '0', 'y5': '0'}
    {'X0': 0.437881, 'X2': 0.818882}
    {'y0': '1', 'y1': '0', 'y2': '0', 'y3': '1', 'y4': '0', 'y5': '0'}
    {'X0': 0.676477, 'X1': 0.724635, 'X2': 0.755123}
    {'y0': '0', 'y1': '0', 'y2': '1', 'y3': '0', 'y4': '0', 'y5': '0'}

    This function can also deal with missing features in non-sparse data. These are indicated with
    a question mark.

    >>> data = '''
    ... @relation giveMeLoan-weka.filters.unsupervised.attribute.Remove-R1
    ... @attribute RevolvingUtilizationOfUnsecuredLines numeric
    ... @attribute age numeric
    ... @attribute NumberOfTime30-59DaysPastDueNotWorse numeric
    ... @attribute DebtRatio numeric
    ... @attribute MonthlyIncome numeric
    ... @attribute NumberOfOpenCreditLinesAndLoans numeric
    ... @attribute NumberOfTimes90DaysLate numeric
    ... @attribute NumberRealEstateLoansOrLines numeric
    ... @attribute NumberOfTime60-89DaysPastDueNotWorse numeric
    ... @attribute NumberOfDependents numeric
    ... @attribute isFraud {0,1}
    ... @data
    ... 0.213179,74,0,0.375607,3500,3,0,1,0,1,0
    ... 0.305682,57,0,5710,?,8,0,3,0,0,0
    ... 0.754464,39,0,0.20994,3500,8,0,0,0,0,0
    ... 0.116951,27,0,46,?,2,0,0,0,0,0
    ... 0.189169,57,0,0.606291,23684,9,0,4,0,2,0
    ... '''

    >>> with open('data.arff', mode='w') as f:
    ...     _ = f.write(data)

    >>> for x, y in stream.iter_arff('data.arff', target='isFraud'):
    ...     print(len(x))
    10
    9
    10
    9
    10

    References
    ----------
    [^1]: [ARFF format description from Weka](https://waikato.github.io/weka-wiki/formats_and_processing/arff_stable/)

    """
    # If a file is not opened, then we open it
    if isinstance(filepath_or_buffer, (str, os.PathLike)):
        buffer = utils.open_filepath(filepath_or_buffer, compression, encoding=encoding)
        should_close = True
    else:
        buffer = filepath_or_buffer
        should_close = False

    try:
        _rel, attrs = read_header(buffer)
    except ValueError as e:
        msg = f"Error while parsing header, error was: {e}"
        raise scipy.io.arff.ParseArffError(msg)

    names = [attr.name for attr in attrs]
    n_attrs = len(names)
    # HACK: it's a bit hacky to rely on class name to determine what casting to apply
    casts = [float if attr.__class__.__name__ == "NumericAttribute" else None for attr in attrs]
    nominal_values = [set(attr.values) if attr.type_name == "nominal" else None for attr in attrs]
    target_names = {target} if isinstance(target, str) else set(target or [])
    defaults = (
        {
            attr.name: attr.values[0] if attr.type_name == "nominal" else 0.0
            for attr in attrs
            if attr.name in target_names
        }
        if sparse
        else {}
    )

    for r in buffer:
        if len(r) == 0:
            continue

        x: dict[FeatureName, typing.Any]
        y: typing.Any | dict[FeatureName, typing.Any] | None

        # Read row
        if sparse:
            x = {}
            row = r.strip()
            if not row or row.startswith("%"):
                continue
            if row[0] != "{" or row[-1] != "}":
                raise ValueError
            entries = row[1:-1].strip()
            sparse_entries = iter(entries.split(",") if entries else [])
            for entry in sparse_entries:
                name_index, val = entry.split(maxsplit=1)
                index = int(name_index)
                if not 0 <= index < n_attrs:
                    raise ValueError
                name = names[index]
                cast = casts[index]
                if cast is not None:
                    if val[0] == "?" and val.strip() == "?":
                        if name in target_names:
                            x[name] = None
                    else:
                        x[name] = cast(val)
                    continue

                val = val.strip()
                if val == "?":
                    if name in target_names:
                        x[name] = None
                    continue

                allowed_values = nominal_values[index]
                if allowed_values is not None and val in allowed_values:
                    x[name] = val
                    continue

                quoted = val[0] in "'\""
                if quoted:
                    quote = val[0]
                    while not val.rstrip().endswith(quote):
                        try:
                            val += "," + next(sparse_entries)
                        except StopIteration as error:
                            raise ValueError from error
                    val = _parse_quoted_value(val.rstrip())
                else:
                    if " " in val or "\t" in val or "'" in val or '"' in val:
                        raise ValueError
                if allowed_values is not None and val not in allowed_values:
                    raise ValueError
                x[name] = val
        else:
            row = r.strip()
            if not row or row.startswith("%"):
                continue
            x = {}
            for name, cast, val in zip(names, casts, _parse_dense_row(row, n_attrs)):
                if val == "?":
                    continue
                if not val:
                    raise ValueError
                if val[0] in "'\"":
                    val = _parse_quoted_value(val)
                elif cast is None and (" " in val or "\t" in val or "'" in val or '"' in val):
                    raise ValueError
                x[name] = cast(val) if cast else val

        # Handle target
        if target is None:
            y = None
        elif isinstance(target, str):
            y = x.pop(target, defaults.get(target)) if target else None
        else:
            y = {name: x.pop(name, defaults.get(name)) for name in target}

        yield x, y

    # Close the file if we opened it
    if should_close:
        buffer.close()
