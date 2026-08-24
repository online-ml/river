from __future__ import annotations

import csv
import io
import random
import typing
from datetime import datetime
from functools import partial

import pytest

from river import stream
from river.stream.iter_csv import DictReader

if typing.TYPE_CHECKING:
    import pathlib
    from collections.abc import Callable

    from river import base
    from river.base.typing import FeatureName
    from tests.stream.conftest import CompressionCase, WriteFile

    Row = tuple[dict[FeatureName, typing.Any], typing.Any]
    MakeStream = Callable[[typing.TextIO], base.typing.Stream]
    Converters = dict[FeatureName, Callable[[typing.Any], typing.Any]]

CONTENT = "name,year,rating\na,2016,9.5\nb,2006,9.4\nc,2001,9.4\nd,2008,9.1\ne,2019,9.0\n"
HOLES = "col1,col2,col3\n,1,2\n5,,4\n3,1,"


def int_or_none(s: str) -> int | None:
    try:
        return int(s)
    except ValueError:
        return None


FIRST_ROW_CASES: dict[str, tuple[MakeStream, Row]] = {
    "defaults": (
        stream.iter_csv,
        ({"name": "a", "year": "2016", "rating": "9.5"}, None),
    ),
    "target": (
        partial(stream.iter_csv, target="rating"),
        ({"name": "a", "year": "2016"}, "9.5"),
    ),
    "targets": (
        partial(stream.iter_csv, target=["year", "rating"]),
        ({"name": "a"}, {"year": "2016", "rating": "9.5"}),
    ),
    "drop": (partial(stream.iter_csv, drop=["year"]), ({"name": "a", "rating": "9.5"}, None)),
    "parse_dates": (
        partial(stream.iter_csv, parse_dates={"year": "%Y"}),
        ({"name": "a", "year": datetime(2016, 1, 1), "rating": "9.5"}, None),
    ),
    # Given field names, the header is data like any other row.
    "fieldnames": (
        partial(stream.iter_csv, fieldnames=["A", "B", "C"]),
        ({"A": "name", "B": "year", "C": "rating"}, None),
    ),
}

CONVERTER_CASES: dict[str, tuple[bool, list[Row]]] = {
    "keep-nones": (
        False,
        [
            ({"col1": None, "col2": 1, "col3": 2}, None),
            ({"col1": 5, "col2": None, "col3": 4}, None),
            ({"col1": 3, "col2": 1, "col3": None}, None),
        ],
    ),
    "drop-nones": (
        True,
        [
            ({"col2": 1, "col3": 2}, None),
            ({"col1": 5, "col3": 4}, None),
            ({"col1": 3, "col2": 1}, None),
        ],
    ),
}


def test_dict_reader_can_be_instantiated() -> None:
    reader = DictReader(fraction=1, rng=random.Random(42), f=io.StringIO(CONTENT))

    assert next(reader) == {"name": "a", "year": "2016", "rating": "9.5"}


@pytest.mark.parametrize(("make", "expected"), FIRST_ROW_CASES.values(), ids=FIRST_ROW_CASES)
def test_the_options_shape_the_rows(make: MakeStream, expected: Row) -> None:
    assert next(make(io.StringIO(CONTENT))) == expected


@pytest.mark.parametrize(("drop_nones", "expected"), CONVERTER_CASES.values(), ids=CONVERTER_CASES)
def test_values_are_cast_by_the_converters(drop_nones: bool, expected: list[Row]) -> None:
    converters: Converters = {"col1": int_or_none, "col2": int_or_none, "col3": int_or_none}

    dataset = stream.iter_csv(io.StringIO(HOLES), converters=converters, drop_nones=drop_nones)
    assert list(dataset) == expected


@pytest.mark.parametrize("content", ["", "name,year,rating\n"], ids=["empty", "header-only"])
def test_a_file_without_rows_yields_nothing(content: str) -> None:
    assert list(stream.iter_csv(io.StringIO(content))) == []


def test_sampling_is_deterministic_for_a_given_seed() -> None:
    sampled = [x["name"] for x, _ in stream.iter_csv(io.StringIO(CONTENT), fraction=0.5, seed=42)]

    assert sampled == [
        x["name"] for x, _ in stream.iter_csv(io.StringIO(CONTENT), fraction=0.5, seed=42)
    ]
    assert set(sampled) <= {"a", "b", "c", "d", "e"}


def test_the_field_size_limit_is_restored(tmp_path: pathlib.Path, write_file: WriteFile) -> None:
    limit = csv.field_size_limit()

    _ = list(stream.iter_csv(write_file(tmp_path / "data.csv", CONTENT), field_size_limit=10**6))

    assert csv.field_size_limit() == limit


def test_a_dataset_is_read_from_a_path(
    tmp_path: pathlib.Path, write_file: WriteFile, compression_case: CompressionCase
) -> None:
    path = write_file(
        tmp_path / f"data.csv{compression_case.suffix}", CONTENT, compression_case.written_as
    )

    dataset = stream.iter_csv(path, target="rating", compression=compression_case.read_as)
    assert next(dataset) == ({"name": "a", "year": "2016"}, "9.5")


def test_a_file_the_reader_opened_is_closed(
    tmp_path: pathlib.Path,
    write_file: WriteFile,
    compression_case: CompressionCase,
    opened_files: list[typing.TextIO],
) -> None:
    path = write_file(
        tmp_path / f"data.csv{compression_case.suffix}", CONTENT, compression_case.written_as
    )

    _ = list(stream.iter_csv(path, compression=compression_case.read_as))

    assert len(opened_files) == 1
    assert opened_files[0].closed


def test_a_buffer_passed_in_by_the_caller_is_left_open() -> None:
    buffer = io.StringIO(CONTENT)

    _ = list(stream.iter_csv(buffer))

    assert not buffer.closed
