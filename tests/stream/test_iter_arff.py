from __future__ import annotations

import io
import typing

import pytest
import scipy.io.arff

from river import stream

if typing.TYPE_CHECKING:
    import pathlib

    from river.base.typing import FeatureName
    from tests.stream.conftest import CompressionCase, WriteFile

    Row = tuple[dict[FeatureName, typing.Any], typing.Any]

DENSE = """@relation test
@attribute a numeric
@attribute b numeric
@attribute c {x,y}
@data
1,2,x
3,?,y
"""

SPARSE = """@relation test
@attribute y0 {0,1}
@attribute y1 {0,1}
@attribute X0 numeric
@attribute X1 numeric
@data
{ 1 1,2 0.5 }
{ 0 1,3 0.25 }
"""

# The second row holds a `?`, so `b` is expected to be missing from its features.
UNTOUCHED: list[Row] = [({"a": 1.0, "b": 2.0, "c": "x"}, None), ({"a": 3.0, "c": "y"}, None)]
DENSE_CASES: dict[str, tuple[str | list[str] | None, list[Row]]] = {
    "target": ("c", [({"a": 1.0, "b": 2.0}, "x"), ({"a": 3.0}, "y")]),
    "no-target": (None, UNTOUCHED),
    "absent-target": ("nope", UNTOUCHED),
    "targets": (
        ["b", "c"],
        [({"a": 1.0}, {"b": 2.0, "c": "x"}), ({"a": 3.0}, {"b": 0, "c": "y"})],
    ),
}


@pytest.mark.parametrize(("target", "expected"), DENSE_CASES.values(), ids=DENSE_CASES)
def test_numeric_attributes_are_cast_and_the_target_is_split_off(
    target: str | list[str] | None, expected: list[Row]
) -> None:
    assert list(stream.iter_arff(io.StringIO(DENSE), target=target)) == expected


def test_sparse_rows_only_carry_the_features_they_list() -> None:
    dataset = stream.iter_arff(io.StringIO(SPARSE), target=["y0", "y1"], sparse=True)

    assert list(dataset) == [
        ({"X0": "0.5"}, {"y0": 0, "y1": "1"}),
        ({"X1": "0.25"}, {"y0": "1", "y1": 0}),
    ]


def test_a_dataset_is_read_from_a_path(
    tmp_path: pathlib.Path, write_file: WriteFile, compression_case: CompressionCase
) -> None:
    path = write_file(
        tmp_path / f"data.arff{compression_case.suffix}", DENSE, compression_case.written_as
    )

    dataset = stream.iter_arff(path, target="c", compression=compression_case.read_as)
    assert list(dataset) == DENSE_CASES["target"][1]


def test_a_file_the_reader_opened_is_closed(
    tmp_path: pathlib.Path,
    write_file: WriteFile,
    compression_case: CompressionCase,
    opened_files: list[typing.TextIO],
) -> None:
    path = write_file(
        tmp_path / f"data.arff{compression_case.suffix}", DENSE, compression_case.written_as
    )

    _ = list(stream.iter_arff(path, target="c", compression=compression_case.read_as))

    assert len(opened_files) == 1
    assert opened_files[0].closed


def test_a_buffer_passed_in_by_the_caller_is_left_open() -> None:
    buffer = io.StringIO(DENSE)

    _ = list(stream.iter_arff(buffer, target="c"))

    assert not buffer.closed


def test_an_unparsable_header_is_reported_as_a_parse_error() -> None:
    dataset = stream.iter_arff(io.StringIO("@relation test\n@attribute a bogus\n@data\n1\n"))

    with pytest.raises(scipy.io.arff.ParseArffError):
        next(dataset)
